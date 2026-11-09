from contextlib import ExitStack
import pytest
from scripts import k5_joint_source as joint


def pair(candidate, state='excluded', reason='cached_nonwriter:x'):
    return [dict(query='q',candidate=candidate,anchor='a',model=m,state=state,reason=reason)
            for m in ('fp32_input','bf16_input')]


def test_resume_preserves_evidence_but_retries_only_missing_inputs():
    saved=pair('good')+pair('missing','unresolved','no weights')+pair('stat','unresolved','no_complete_informative_shared_witness')
    reused=joint.reusable_pairs(saved)
    assert set(reused)=={('q','good'),('q','stat')}
    assert reused[('q','good')]==saved[:2]
    assert reused[('q','stat')]==saved[-2:]


def test_resume_rejects_partial_or_duplicate_hypotheses():
    for saved in [pair('x')[:1],pair('x')+pair('x'),[dict(x,model='invalid') for x in pair('x')]]:
        with pytest.raises(ValueError): joint.reusable_pairs(saved)


def test_buffered_nonsequential_reads_and_truncation(tmp_path):
    p=tmp_path/'bytes';p.write_bytes(bytes(range(100)))
    readers={}
    with ExitStack() as stack:
        assert joint.read_range(p,70,80,readers,stack)==bytes(range(70,80))
        assert joint.read_range(p,1,4,readers,stack)==bytes(range(1,4))
        assert len(readers)==1
        with pytest.raises(joint.Unsupported): joint.read_range(p,98,103,readers,stack)
    assert all(f.closed for f in readers.values())


def test_resume_provenance_rejects_changed_frozen_driver(tmp_path):
    import hashlib,json
    code=b'original frozen driver'
    (tmp_path/'driver.py').write_bytes(code)
    (tmp_path/'protocol.json').write_text(json.dumps(dict(driver_sha256=hashlib.sha256(code).hexdigest())))
    progress=tmp_path/'progress.json';progress.write_text('{"pairs": []}')
    value=joint.resume_provenance(progress)
    assert value['driver_sha256']==hashlib.sha256(code).hexdigest()
    (tmp_path/'driver.py').write_bytes(b'changed')
    with pytest.raises(ValueError): joint.resume_provenance(progress)
