"""Report which gated Hugging Face repositories the current token can read.

Reads a CSV with a ``repo`` column and writes the same rows plus an
``access`` column: ``granted``, ``no_access``, ``missing`` or ``error``.
The token comes from ``HF_TOKEN``; it is never printed or written.

    HF_TOKEN=... python scripts/check_hf_access.py \
        docs/access/gated_repos.csv docs/access/access_status.csv
"""

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

HUB = "https://huggingface.co"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def request(url, token, method="GET"):
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}", "User-Agent": "model-zoo-access-check"})
    try:
        with OPENER.open(req, timeout=30) as response:
            return response.status, response.read() if method == "GET" else b""
    except urllib.error.HTTPError as err:
        return err.code, b""


def probe_file(repo, token):
    """Pick a non-card file; the model card stays public on gated repos."""
    status, body = request(f"{HUB}/api/models/{repo}", token)
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"metadata status {status}")
    names = [s["rfilename"] for s in json.loads(body).get("siblings", [])]
    for preferred in ("config.json", "model.safetensors.index.json"):
        if preferred in names:
            return preferred
    others = [n for n in names if n not in ("README.md", ".gitattributes")]
    return others[0] if others else None


def check(repo, token):
    try:
        filename = probe_file(repo, token)
        if filename is None:
            return "missing"
        status, _ = request(f"{HUB}/{repo}/resolve/main/{filename}", token, method="HEAD")
    except (RuntimeError, urllib.error.URLError, TimeoutError) as err:
        print(f"{repo}: {err}", file=sys.stderr)
        return "error"
    if status in (200, 302, 307):
        return "granted"
    if status in (401, 403):
        return "no_access"
    return "missing" if status == 404 else "error"


def main(source, destination):
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN in the environment first.")
    with open(source, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["access"] = check(row["repo"], token)
        print(f"{row['access']:>9}  {row['repo']}")
        time.sleep(0.5)
    with open(destination, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["access"]] = counts.get(row["access"], 0) + 1
    print(counts)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
