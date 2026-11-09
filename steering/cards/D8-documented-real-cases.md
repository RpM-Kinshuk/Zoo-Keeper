# D8: Count real Hub models as test cases when their construction is published

Status: decided (overseer, Nov 9: "go ahead" on the check-in corrections)

**Decision.** A real Hub model counts as **documented truth** when whoever made
it published how it was made:
- a vendor's own quantized or FP8 release of its own model, with the method
  stated;
- a merge whose repository ships its `mergekit_config.yml`;
- training stages released by the developer (for example OLMo-2 and Tülu 3);
- a refusal removal made with a public tool whose card records the base model
  and the settings.

This is the "verified Hub" tier the plan always intended.

**What it means in practice:**
- Test the exact marks on these cases using the published recipe.
- If the mark is not found, that is a real result ("mark not found, under the
  documented recipe"), not an eligibility gap.
- Unknown details that the recipe does not pin (for example an export path or
  a random seed) are listed as caveats. They do not block the test.
- Cases with only a declared relation and no published recipe stay in the
  **declared** tier. They are reported separately as description, and never
  block work.
- No contact with model makers is needed or allowed without a separate
  decision.
- D7 still applies: naming a unique parent requires telling relatives apart
  within the stated pool.

**Why:** without real cases, the headline ("exact marks beat published
methods") holds only inside our lab, and requiring makers' private records had
stalled all three tracks.

**Big-picture effect:**
- Claims: makes C2's real-model check possible.
- Papers and dates: protects the Dec 12 method check and the Dec 18 venue
  call.
- Real-world use: these are exactly the cases users ask about.
- Cost: small, and under the 100-model limit.
- Risk: some marks may blur on real models. That would be reported as a
  limit.

**Can it be undone?** Yes.
