"""Headless loader check: ``python -m tsda.cli path/to/data.csv``.

Runs the same :func:`~tsda.loader.load_timeseries` the Streamlit UI uses and
prints the resulting shape + dtypes. Handy for verifying a file parses
without starting the browser app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import LoadError, load_timeseries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsda", description="Load a CSV/JSON time-series file")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    try:
        frame = load_timeseries(args.path.name, args.path.read_bytes())
    except FileNotFoundError:
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 1
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"rows x cols : {frame.shape[0]} x {frame.shape[1]}")
    print("columns     :", list(frame.columns))
    print("dtypes      :", {c: str(d) for c, d in frame.dtypes.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
