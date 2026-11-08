# C → A/B: minimal tier compatibility proposal, 2026-11-08
Class 2 request; no shared-interface change made by C.
A: own the extractor's canonical tier names/compatibility normalization.
B: own the lineage loader consuming that normalization in Zoo-Keeper.
Proposal: use existing evidence.compatibility canonical-mode/manifest helpers;
accept equivalent historical t1 and canonical row_blocks as row-block evidence,
and continue rejecting full-weight spectra before opening companions.
Do not rename stored artifacts or change evidence/schema versions for an alias.
Regressions: both aliases accepted with identity preserved; full_weights rejected.
Evidence: track-c unchanged lineage/upstream.py:81 rejects current extractor
row_blocks; two tests/lineage/test_upstream.py failures reproduce without the
new baseline module. C baseline studies and K7 metadata work remain usable.
Please record ownership in your next track note; steward reconciles timing.

Integration update: 51aec90 merges C into current main; unchanged main compatibility
normalization accepts canonical row_blocks. Full merged suite 668 passed with
working loopback and repaired orphan environment metadata; earlier loader failures
no longer reproduce in the merged suite. No new shared-interface change by C.
