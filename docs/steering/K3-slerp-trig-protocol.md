# K3 exact trigonometric follow-up — before execution

The endpoint audit rejected a uniform linear-fallback assumption for Qwen2.5.
Keep that negative result. Now test the actual standard SLERP formula: for each
tensor parent cosine d, linear coefficients [1-t,t] if abs(d)>0.9995; otherwise
[sin((1-t)acos(d))/sin(acos(d)), sin(t acos(d))/sin(acos(d))].
Read failed endpoint tensors anew and check child reproduction under the SAME
8 float32 epsilon * max(1,max abs expected) tolerance. This is an empirical
standard-formula check, not proof of the historical mergekit source hash.

Compare prior fitted parent coefficients against these exact per-tensor targets:
constant-t uses its existing all-view fit; layerwise uses the existing matching
layer fit, excludes nonlayer tensors from coefficient evaluation (but still
checks their child reproduction). Include only views actually in the fit.
PASS case iff maximum absolute coefficient discrepancy across evaluated
parent coefficients and eligible tensors <=0.05. Combined linear/SLERP target
remains <=4 misses. No new fit, no truth, threshold or split change.
