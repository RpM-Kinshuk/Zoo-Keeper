from concurrent.futures import ThreadPoolExecutor
import pytest


def test_budget_is_shared_across_threads_and_failed_transport_attempts():
    from scripts.task_choice_download import ResponseBudget, AcquisitionStopped
    budget = ResponseBudget(used=20, limit=30)
    def reserve(_):
        try:
            budget.reserve(3)
            return True
        except AcquisitionStopped:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(reserve,range(10)))
    assert sum(result) == 3 and budget.reserved == 29
    with pytest.raises(AcquisitionStopped):
        budget.reserve(2)
    assert budget.reserved == 29


def test_parallel_append_resumes_prefix_in_order_and_verifies_full_hash(tmp_path):
    from scripts.task_choice_download import append_ranges
    import hashlib
    content = b'abcdef0123456789'
    p = tmp_path/'partial';p.write_bytes(content[:3])
    seen = []
    def fetch(start,end):
        seen.append((start,end))
        return content[start:end], {'bytes_read':end-start,'range_reads':1}
    stats = append_ranges(p,len(content),fetch,chunk=4,workers=3)
    assert p.read_bytes() == content
    assert min(s for s,e in seen) == 3
    assert stats['bytes_read'] == len(content)-3
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(content).hexdigest()
