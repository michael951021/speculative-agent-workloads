"""CLI: python -m draftlab.data [build | stats]."""

import argparse

from .corpus import build, stats


def main() -> None:
    ap = argparse.ArgumentParser(prog="draftlab.data")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build", help="build data/corpus/*.parquet from data/raw/agent_turns.jsonl")
    sub.add_parser("stats", help="run data/stats.sql -> results/corpus_stats.json")
    a = ap.parse_args()
    if a.cmd == "stats":
        stats()
    else:
        build()


if __name__ == "__main__":
    main()
