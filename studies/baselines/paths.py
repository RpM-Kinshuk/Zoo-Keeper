"""Locate the shared checkout layout from main or a track worktree."""
from pathlib import Path


def workspace_root(repo_root):
    for candidate in Path(repo_root).resolve().parents:
        if (candidate/'Zoo-Keeper').is_dir() and (candidate/'Model-Zoo').is_dir():
            return candidate
    raise FileNotFoundError('Expected sibling Zoo-Keeper and Model-Zoo checkouts')
