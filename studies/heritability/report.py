"""Atomic, portable result reports for the frozen study runner."""

import os
from pathlib import Path
import tempfile

import pandas as pd


def atomic_write(path, content):
    """Publish complete bytes through a temporary sibling and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode('utf-8') if isinstance(content, str) else content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _table(frame, columns):
    columns = [name for name in columns if name in frame]
    rows = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for record in frame[columns].to_dict('records'):
        rows.append('| '+' | '.join('unavailable' if pd.isna(record[name]) else str(record[name]).replace('|', '/') for name in columns)+' |')
    return '\n'.join(rows)


def write_report(run, coverage, diagnostics, heritability, parents, exchange, skill_errors=None):
    """Write report.md last; exceptions preserve a previously completed report."""
    join = coverage.get('score_to_model_join', {})
    prediction_join = coverage.get('prediction_to_score_join', {})
    text = [f"# Heritability first pass — {coverage['status']}", '',
        'Familial fractions describe the specified covariance model; they are not causal heritability claims.', '',
        f"Observed scores: {coverage.get('retained_observations', 0)}; primary deployments: {coverage.get('primary_observations', 0)}; known sampling variances: {coverage.get('variance_available', 0)}.",
        f"Verified score-to-model joins: {join.get('numerator', 0)}/{join.get('denominator', 0)} (denominator: retained deployment/task observations).",
        f"Verified prediction-to-score joins: {prediction_join.get('numerator', 0)}/{prediction_join.get('denominator', 0)} (denominator: all input prediction rows).",
        f"Ancestry status: {coverage.get('ancestry_status', 'unavailable')}; historical status: {coverage.get('historical_status', 'unavailable')}.", '',
        '## Familial variance', '',
        _table(heritability, ['benchmark','task','harness','relation_tier','scale','covariance_model','n','families','status','reason','converged','boundary','familial_fraction','measurement_variance_fraction','noise_ceiling','familial_fraction_lower','familial_fraction_upper']), '',
        'Noise ceiling is the measurement-reliability ceiling for R²; known sampling noise is reported separately from the latent familial fraction.', '',
        '## Observed-parent regression', '',
        _table(parents, ['benchmark','task','harness','scale','n','n_corrected','naive_slope','corrected_slope','variance_denominator','error_assumption','status','reason']), '',
        '## Exchange rates', '',
        _table(exchange, ['benchmark','task','estimand','method','analysis_tier','status','reason','k','incremental_k','family_stratum']), '',
        'Missing rows and unsuccessful estimates remain unavailable with reasons. Bootstrap percentile intervals require at least 90% successful replicates; diagnostics retain failed counts. Raw and corrected-binomial logit fits use their reported support.', '',
        '## Per-skill prediction errors', '',
        _table(skill_errors if skill_errors is not None else pd.DataFrame(), ['benchmark','task','harness','precision','quantization','setting','fold','method','analysis_tier','n_observed','n_audited','n','excluded_from_common_support','mae','rmse','mae_sd','r2','status','reason']), '',
        'Scale-free comparisons use the same audited checkpoint intersection across supported tasks and methods within each deployment/holdout stratum. Full per-task descriptive errors and excluded support remain in skill_errors.csv. Constant targets leave MAE/sd and R² unavailable.', '',
        '## Diagnostics', '',
        f"Convergence failures: {diagnostics.get('convergence_failures', 0)}. Requested bootstrap replicates: {diagnostics.get('bootstrap_replicates', 0)}. Seed: {diagnostics.get('seed')}.",
        'Protocol and input hashes are verified before every analysis or resume. No predictor is refitted by this runner.', '']
    atomic_write(Path(run)/'report.md', '\n'.join(text))
