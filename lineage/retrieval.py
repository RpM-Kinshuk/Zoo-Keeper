"""Exact candidate retrieval for synthetic evidence, separate from calibration."""

import numpy as np

from .records import FingerprintRecord, cosine_views, view_diagnostics


def exclusion_reason(query, candidate):
    """Explain self/duplicate exclusions, including when no rows are returned."""
    if candidate.key == query.key:
        return 'self'
    group = query.meta.get('duplicate_group')
    if group is not None and group == candidate.meta.get('duplicate_group'):
        return 'exact_duplicate_group'
    return None


def retrieve(query: FingerprintRecord, candidates: list[FingerprintRecord], *, limit: int = 50) -> list[dict]:
    """Return eligible candidates; never a calibrated score.

    Self and alternatives in the query's exact duplicate group are excluded.
    Call exclusion_reason and records.view_diagnostics to audit exclusions;
    partial view exclusions also appear in each returned row's reason.
    """
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    if not query.meta.get('synthetic', False):
        from .upstream import retrieve_real
        return retrieve_real(query, candidates, limit=limit)
    results = []
    seen = set()
    for candidate in candidates:
        if candidate.key in seen or exclusion_reason(query, candidate):
            continue
        seen.add(candidate.key)
        if not candidate.meta.get('synthetic', False):
            continue
        # Eligibility is always assessed at the frozen T1 evidence size.
        pairs, reasons = view_diagnostics(query, candidate, k=64)
        retrieval_pairs, retrieval_reasons = view_diagnostics(query, candidate, k=32)
        retrieval_set = set(retrieval_pairs)
        pairs = [pair for pair in pairs if pair in retrieval_set][:32]
        reasons = sorted(set(reasons + retrieval_reasons))
        if not pairs:
            continue
        scores = cosine_views(query, candidate, pairs, k=32)
        # A nonzero 64-square view may have a zero 32-square retrieval block.
        if not np.isfinite(scores).all():
            continue
        results.append({'parent_key': candidate.key, 'retrieval_score': float(np.mean(scores)),
                        'n_shared': len(pairs), 'reason': ';'.join(reasons) if reasons else 'ok'})
    return sorted(results, key=lambda row: (-row['retrieval_score'], row['parent_key']))[:limit]
