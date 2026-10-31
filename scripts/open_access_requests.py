"""Open gated Hugging Face repos one tab at a time; closing a tab opens the next.

You file each access request by hand in the browser. The script never clicks
anything. It watches Chrome's open tabs (via AppleScript) and opens the next
repo once no tab still shows the current one. Progress is logged, so a re-run
continues where you stopped.

    python scripts/open_access_requests.py                 # tiers 1 and 1b
    python scripts/open_access_requests.py --tiers 2        # later, census tier
    python scripts/open_access_requests.py --manual         # press Enter instead

macOS asks once for permission to let the terminal control Chrome; allow it.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ACCESS_DIR = Path(__file__).resolve().parents[1] / "docs/access"
QUEUE = ACCESS_DIR / "request_queue.csv"
LOG = ACCESS_DIR / "request_log.csv"
BROWSER = "Google Chrome"


def open_tab(url):
    subprocess.run(["open", "-a", BROWSER, url], check=True)


def tab_urls():
    """All open tab URLs, or None if Chrome cannot be queried."""
    script = f'tell application "{BROWSER}" to get URL of every tab of every window'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return [unquote(u.strip()).lower() for u in result.stdout.split(",") if u.strip()]


def repo_is_open(repo, urls):
    # Covers the repo page, its sub-pages, and login redirects (?next=/org/repo).
    return any(f"/{repo.lower()}" in u for u in urls)


def wait_until_shown(repo, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        urls = tab_urls()
        if urls is None:
            return False
        if repo_is_open(repo, urls):
            return True
        time.sleep(1)
    return False


def wait_until_closed(repo):
    while True:
        urls = tab_urls()
        if urls is None:
            raise RuntimeError("lost access to Chrome tabs")
        if not repo_is_open(repo, urls):
            return
        time.sleep(1.5)


def done_repos():
    if not LOG.exists():
        return set()
    with open(LOG, newline="") as handle:
        return {row["repo"] for row in csv.DictReader(handle)}


def log(repo, opened_at):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["repo", "opened_at", "closed_at"])
        writer.writerow([repo, opened_at, datetime.now(timezone.utc).isoformat(timespec="seconds")])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="+", default=["1", "1b"])
    parser.add_argument("--manual", action="store_true", help="press Enter for the next repo")
    parser.add_argument("--restart", action="store_true", help="ignore the progress log")
    args = parser.parse_args()

    with open(QUEUE, newline="") as handle:
        queue = [row for row in csv.DictReader(handle) if row["tier"] in args.tiers]
    skip = set() if args.restart else done_repos()
    todo = [row for row in queue if row["repo"] not in skip]
    print(f"{len(todo)} to open ({len(queue) - len(todo)} already done). Ctrl-C stops; re-run resumes.")

    manual = args.manual
    for i, row in enumerate(todo, 1):
        repo, url = row["repo"], f"https://huggingface.co/{row['repo']}"
        opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{i}/{len(todo)}] {row['family']}: {repo}", flush=True)
        open_tab(url)
        if not manual and not wait_until_shown(repo):
            print("  Could not see Chrome's tabs; switching to manual mode (press Enter).")
            manual = True
        if manual:
            input("  Request access, then press Enter for the next repo... ")
        else:
            wait_until_closed(repo)
        log(repo, opened_at)
    print("All done. Run scripts/check_hf_access.py once approvals arrive.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped; re-run to continue.")
