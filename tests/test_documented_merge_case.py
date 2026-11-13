
def test_removed_recipe_ingredient_is_never_reintroduced_as_base():
    from scripts.documented_merge_case import candidate_pools,zoo_inputs
    from types import SimpleNamespace
    records={k:SimpleNamespace(key=k) for k in ['base','instruct','sft','adapter']}
    pools=candidate_pools(list(records),['base','instruct'])
    assert set(pools['present'])==set(records)
    assert set(pools['remove_both'])=={'sft','adapter'}
    for name,keys in pools.items():
        args=zoo_inputs(records,keys)
        assert {r.key for r in args['parents']}|{args['base'].key}==set(keys)
        if name=='remove_base':assert 'base' not in keys
        if name=='remove_instruct':assert 'instruct' not in keys
