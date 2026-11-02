"""Baseline adapter interface; absence of an artifact is never a zero error."""
from abc import ABC, abstractmethod

import pandas as pd

BASELINES = ('Trace', 'Spectral Signatures', 'modelDNA', 'Witness Overlap',
             '2602.00772', 'MoTHer', 'metadata imputation')


class Baseline(ABC):
    @abstractmethod
    def run(self, query_keys: list[str], candidate_keys: list[str], *,
            budget_bytes: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return exact LINEAGE_COLUMNS and per-query measured costs.

        Costs include query_key,bytes_read,seconds,status,reason plus separate
        query_bytes,reference_bytes,verification_bytes and an explicit
        reference_amortization_queries denominator. Adapters belong with the
        baseline implementation and must identify unavailable measurements.
        """


def baseline_reports(config):
    """Only explicit measured baseline results can establish a comparison."""
    supplied = config.get('baseline_results', {})
    result = {}
    for name in BASELINES:
        row = supplied.get(name)
        if not row:
            result[name] = {'status': 'not_run', 'reason': 'no baseline artifact supplied'}
        else:
            # The caller supplies frozen provenance; a bare metric never proves a run.
            if not row.get('artifact_hash') or not row.get('measured'):
                result[name] = {'status': 'not_run', 'reason': 'missing measured artifact provenance'}
            else:
                result[name] = dict(row, status='reported')
    return result
