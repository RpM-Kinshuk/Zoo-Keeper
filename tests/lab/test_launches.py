"""Frozen launch checks reject leakage, unsafe masks and altered evidence."""
import pytest
import torch
from lab.launches import split_ids, preference_row, causal_row, check_files

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text.split()) + 1))

def test_partition_leakage_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        split_ids({'training': ['a'], 'direction': ['a'], 'validation': [], 'scoring': []})

def test_causal_never_pads_short_samples():
    assert causal_row(Tokenizer(), [{'role':'user','content':'short'}], 8) is None
    assert len(causal_row(Tokenizer(), [{'role':'user','content':' '.join(['x']*20)}], 8)) == 8

def test_preference_completion_masks_are_suffixes_with_shared_prompt():
    result = preference_row(Tokenizer(), 'one two', ' '.join(['x']*12), ' '.join(['y']*10), 8)
    chosen,rejected,cm,rm = map(torch.tensor,result)
    assert torch.equal(chosen[:2],rejected[:2])
    assert cm.tolist() == rm.tolist() == [0,0,1,1,1,1,1,1]
    assert preference_row(Tokenizer(),'one two','short','short',8) is None

def test_preflight_rejects_changed_file(tmp_path):
    (tmp_path/'evidence').write_text('altered')
    with pytest.raises(ValueError,match='hash mismatch'):
        check_files(tmp_path, {'evidence':'0'*64})

def test_unresolved_dpo_parent_cannot_be_resolved_without_completed_output(tmp_path):
    from lab.launches import resolve_sft_parent
    with pytest.raises(ValueError, match='completed SFT'):
        resolve_sft_parent({'trainer':'dpo','recipe':{'parents':['missing-sft']}}, tmp_path)

def test_pinned_publisher_license_must_match_claim(tmp_path):
    import hashlib
    from lab.launches import check_license
    card=tmp_path/'README.md';card.write_text('---\nlicense: unknown\n---\n')
    pin={'license_path':'README.md','license_sha256':hashlib.sha256(card.read_bytes()).hexdigest(),'license':'mit','license_evidence_type':'pinned_readme_declaration'}
    with pytest.raises(ValueError,match='license declaration'):
        check_license(tmp_path,pin)
