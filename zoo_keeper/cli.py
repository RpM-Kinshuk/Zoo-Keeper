"""Offline query commands: python -m zoo_keeper.cli COMMAND ..."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import duckdb
import pandas as pd

from . import AmbiguousModel, NotComparable, UnknownModel, Zoo


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("expected a finite number")
    return number


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data_root = os.environ.get("ZOO_DATA")
    default_zoo = Path(data_root) / "builds" / "current" if data_root else None
    for command in ("find", "info", "similar", "family", "diff", "check"):
        sub = commands.add_parser(command)
        sub.add_argument("--zoo", type=Path, default=default_zoo,
                         help="zoo directory (default: $ZOO_DATA/builds/current)")
        sub.add_argument("--data_root", type=Path,
                         help="original evidence root (default: $ZOO_DATA)")
        sub.add_argument("--json", action="store_true", help="emit one JSON document")
        if command == "find":
            sub.add_argument("--modality")
            sub.add_argument("--license", nargs="+")
            sub.add_argument("--params_min", type=_finite_float)
            sub.add_argument("--params_max", type=_finite_float)
            sub.add_argument("--format", dest="weight_format")
            sub.add_argument("--access")
            sub.add_argument("--text")
            sub.add_argument("--skill")
            sub.add_argument("--min_score", type=_finite_float)
            sub.add_argument("--include_unverified", dest="verified_only", action="store_false")
            sub.add_argument("--decision", action="store_const", const=True)
            sub.add_argument("--system_one", action="store_const", const=True)
            sub.add_argument("--architecture")
            sub.add_argument("--limit", type=int, default=50)
        elif command == "diff":
            sub.add_argument("a", metavar="A")
            sub.add_argument("b", metavar="B")
        else:
            sub.add_argument("model", metavar="MODEL")
        if command == "similar":
            sub.add_argument("-k", type=int, default=10)
            sub.add_argument("--keep_duplicates", dest="collapse_duplicates", action="store_false")
        elif command == "family":
            sub.add_argument("--depth", type=int, default=2)
        if command in ("diff", "check"):
            sub.add_argument("--null", type=Path, help="empirical calibration JSON")
    return parser


def main(argv=None):
    """Print a query result and return its process exit status."""
    parser = _parser()
    try:
        options = vars(parser.parse_args(argv))
        path = options.pop("zoo")
        if path is None:
            parser.error("--zoo is required when ZOO_DATA is unset")
    except SystemExit as error:
        return error.code

    command = options.pop("command")
    as_json = options.pop("json")
    data_root = options.pop("data_root")
    try:
        with Zoo(path, data_root=data_root) as opened:
            result = getattr(opened, command)(**options)
        if isinstance(result, pd.DataFrame):
            output = result.to_json(orient="records", indent=2, double_precision=15) if as_json else result.to_string(index=False)
        else:
            output = json.dumps(result, indent=2, allow_nan=False)
    except (UnknownModel, AmbiguousModel, NotComparable, OSError, ValueError, duckdb.IOException) as error:
        print(f"{command}: {error}", file=sys.stderr)
        return 2
    print(output)
    return 1 if command == "check" and result.get("verdict") == "inconsistent" else 0


if __name__ == "__main__":
    raise SystemExit(main())
