# Witness Overlap: exploratory ordinary-fine-tune orientation
Predeclared before weights/scores, 2026-11-08; C/K2, supporting B/K9.

Scientific purpose: test whether a third checkpoint provides useful orientation
for known ordinary-fine-tune edges, which symmetric relatedness cannot orient.
Existing spectra/sketches cannot recover full delta dot products. Reuse B's
sixteen constructed SFT/DPO edges and existing weights; no new models or training.
Source: arxiv2609.31784v2 equations1,5 and Algorithm1, stored under
runs/track-c-k2-remaining-20261108/witness/. Official code unavailable; label this
an equation-level cosine adaptation, not the full upstream implementation.

Fixed inputs: proposed-triples.json from that audit, all16 edges unchanged,
opposite-seed SFT witness per edge. Four anchors, five checkpoints each,20models.
Witness is distinct and shares inherited coordinates by construction. SFT seeds
share training choices/data, so witnesses are not claimed independent. This is
exploratory, already exposed lab data; no new split/truth/scientific threshold.

Use every complete 2D q/k/v/o/up/down/gate projection tensor at every layer.
Require exactly the seven roles × config depth, matching keys/shapes; floating storage dtypes may differ (see amendment).
Exclude embeddings, LM head,1D norms/biases explicitly; log exclusions.
Float64 decoded stored weights; stream128rows, accumulate full-tensor
u=B-A,v=W-A,t=W-B squared norms and u·v,(-u)·t. Equation1 scores both anchors.
First average layer scores within projection role, then seven equal role means
(Algorithm1 block-aware mean). Lower mean predicts parent; exact tie abstains.
Use same eligible tensor subset for both endpoints. Any zero outgoing norm
makes both anchor cosines undefined: exclude that tensor with reason/count;
if any role has no eligible layer, abstain. Report coverage, never hide exclusions.
Nonfinite input, nonfloating projection or missing key/shape mismatch: not_run.
No arbitrary epsilon/noise cutoff fitted; these are numerical cosines on stored
values, not calibrated physical direction detection. Tiny deltas remain a limit.

Necessary cheap interpretation control: squared distance to the witness from
both endpoints, summed across the same eligible tensors; closer predicts parent.
For a single tensor cosine orientation agrees with this distance rule (triangle
geometry); test that algebraically. Aggregation can differ. This is a diagnostic,
not a new claimed method or calibrated parent-search result. No extra weight reads.
No missing-parent, edge-type, causal order or uncertainty guarantee inferred.

Operational PASS iff all16 cases are accounted for and pins match; abstentions
and not_run are explicit and never count correct. Report correct/answered/eligible,
wrong/tie/unresolved, per-family and SFT/DPO counts, margins, tensor coverage and
full read/hash costs. No scientific pass threshold: current K9 is exploratory.
Algebra tests PASS at float64 tolerance1e-10 for known orthogonal branches,
endpoint-swap equivariance, streaming equivalence, exact ties/zero norms,
role weighting and cosine/distance single-tensor agreement; invalid data rejected.
Write and run failing tests first; one core review; only baseline package checks.

CPU8threads,≤20 local models,expected<30min,stop110min; no GPU reservation or
network weights. Reuse safetensors.safe_open and lab.artifacts.atomic_json.
No raw blocks/deltas retained; scalar results only. Verify weights against B's
existing hash record; bind recipes/triples/source and driver in one input record.
SVD deferred: cosine answers this initial question without singular-vector cost;
only add it for a named failure or the required final coverage, not GPU occupancy.

## Operational amendment before first orientation score
Initial attempts stopped during metadata/type validation, before scores. Training
recipes use inputs.local_parent/model_pin, unlike the derived-quantizer schema.
The OLMo base is float32 and descendants BF16. Equal dtype was an unnecessary
restriction absent from the paper: compare exact stored values decoded to float64,
record the difference rather than excluding valid inputs. This is a documented
implementation/eligibility correction, no label/split/calibration threshold change.

Necessary rounding control: also compute the SAME frozen triples/roles/algorithm
on a shared BF16 view of all three endpoints (convert each row to BF16 then
float64). Primary stored-weight scores remain unchanged. This secondary analysis
asks whether removing unequal storage precision changes orientation or coverage;
it is not a new construction edge or a noise-corrected detector. Record every
zero-delta exclusion and abstention in each mode separately. No extra weight IO.
Report16 primary cases plus16 control cases; compare predictions/coverage before
claiming the direction signal is about training. No guarantee that BF16 controls
remove all numerical confounding. Original failed attempts have no scientific
results; final costs exclude their extra validation/hash IO.
