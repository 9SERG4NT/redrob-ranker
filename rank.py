#!/usr/bin/env python3
"""Redrob Intelligent Candidate Discovery & Ranking - submission entry point.

Reproduce command (Stage-3 compatible):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Accepts plain or gzipped JSONL. Runs CPU-only, no network, well within the
5-minute / 16 GB budget for the 100K-candidate pool.
"""

from __future__ import annotations

import argparse

from redrob_ranker.pipeline import rank


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    ap.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates JSONL (.jsonl or .jsonl.gz).",
    )
    ap.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path for the top-100 ranking.",
    )
    ap.add_argument("--top", type=int, default=100, help="How many candidates to output.")
    args = ap.parse_args()

    rank(args.candidates, args.out, top_n=args.top)


if __name__ == "__main__":
    main()
