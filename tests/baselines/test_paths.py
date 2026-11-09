import pytest

from studies.baselines.paths import workspace_root


@pytest.mark.parametrize('checkout', ['Zoo-Keeper', 'worktrees/Zoo-Keeper-track-c'])
def test_main_and_track_checkout_find_same_workspace(tmp_path, checkout):
    (tmp_path/'Zoo-Keeper').mkdir()
    (tmp_path/'Model-Zoo').mkdir()
    repo = tmp_path/checkout
    repo.mkdir(parents=True, exist_ok=True)
    assert workspace_root(repo) == tmp_path


def test_missing_workspace_fails_without_guessing(tmp_path):
    repo = tmp_path/'standalone'
    repo.mkdir()
    with pytest.raises(FileNotFoundError):
        workspace_root(repo)
