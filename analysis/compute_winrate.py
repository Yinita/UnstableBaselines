#!/usr/bin/env python3
"""
Compute win-rate statistics from eval sample CSVs produced by a run.

Key behavior:
- Scans one or more run directories and aggregates across all files that match: eval_samples_game_*.csv
- Handles mixed/duplicated headers and noisy lines inside a single CSV
- Determines win/loss by using final_reward for rows where pid == eval_model_pid for each game_idx
- Prints overall win rate, per-opponent breakdown, and a rolling win-rate timeline

Usage:
  python analysis/compute_winrate.py --base /path/to/run/logs/samples
  python analysis/compute_winrate.py --base outputs/2025-08-22/00-24-49/.../logs/samples
  python analysis/compute_winrate.py --glob "/abs/path/**/logs/samples"  # multiple runs

If no arguments are provided, the script will try to find the latest run under outputs/ by mtime.
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

NEEDED_COLS = {"game_idx", "pid", "final_reward", "eval_model_pid"}
OPTIONAL_COLS = {"eval_opponent_name"}


def normalize(s: object) -> str:
    return (str(s) if s is not None else "").strip()


def iter_eval_sample_files(bases: List[str]) -> Iterable[str]:
    seen = set()
    for base in bases:
        # Allow base to be a directory that contains the csvs or a glob to directories
        if any(ch in base for ch in "*?[]"):
            dirs = glob.glob(base)
        else:
            dirs = [base]
        for d in dirs:
            # Support both absolute and relative
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                continue
            for p in glob.glob(os.path.join(d, "*.csv")):
                ap = os.path.abspath(p)
                if ap not in seen:
                    seen.add(ap)
                    yield ap


def find_latest_samples_dir(outputs_root: str = "outputs") -> str | None:
    """Find the latest samples directory under outputs by modification time."""
    if not os.path.isdir(outputs_root):
        return None
    candidates = []
    for root, dirs, files in os.walk(outputs_root):
        if os.path.basename(root) == "samples" and os.path.basename(os.path.dirname(root)) == "logs":
            candidates.append(root)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute win-rate from eval sample CSVs")
    ap.add_argument("--base", action="append", default=[], help="Path to a logs/samples directory (can repeat)")
    ap.add_argument("--glob", action="append", default=[], help="Glob to logs/samples directories (can repeat)")
    ap.add_argument("--window", type=int, default=100, help="Rolling window size for timeline")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    bases: List[str] = []
    bases.extend(args.base)
    bases.extend(args.glob)

    if not bases:
        latest = find_latest_samples_dir()
        if latest is None:
            print("No samples directory found. Provide --base /path/to/logs/samples", file=sys.stderr)
            return 1
        print(f"[info] Using latest samples dir: {latest}")
        bases = [latest]

    files = sorted(iter_eval_sample_files(bases))
    if not files:
        print("No eval_samples_game_*.csv files found under given bases.", file=sys.stderr)
        return 1

    # Aggregate last row per game_idx where pid == eval_model_pid
    by_game: Dict[str, Dict[str, str]] = {}

    for p in files:
        try:
            with open(p, "r", newline="") as f:
                header: List[str] | None = None
                for raw in f:
                    line = raw.strip("\n")
                    if not line:
                        continue
                    # Detect header lines that include needed cols (case-insensitive)
                    if "," in line:
                        parts = [c.strip() for c in line.split(",")]
                        lower = {c.lower() for c in parts}
                        if NEEDED_COLS.issubset(lower):
                            header = parts
                            # consume header; next lines will be data rows with these fieldnames
                            continue
                    if header is None:
                        # ignore lines until we see a proper header
                        continue
                    # Parse according to current header
                    reader = csv.DictReader([line], fieldnames=header)
                    for row in reader:
                        gid = normalize(row.get("game_idx"))
                        if not gid:
                            continue
                        pid = normalize(row.get("pid"))
                        evp = normalize(row.get("eval_model_pid"))
                        if evp and pid and pid == evp:
                            by_game[gid] = row
                        else:
                            by_game.setdefault(gid, row)
        except Exception as e:
            print(f"[warn] Failed to parse {p}: {e}", file=sys.stderr)
            continue

    wins = 0
    played = 0
    by_opp = defaultdict(lambda: [0, 0])  # name -> [wins, played]
    timeline: List[Tuple[int, int]] = []

    for gid, row in by_game.items():
        fr = normalize(row.get("final_reward"))
        if fr == "":
            continue
        try:
            val = float(fr)
        except Exception:
            continue
        res = 1 if val > 0 else (0 if val < 0 else None)
        if res is None:
            continue
        played += 1
        wins += res
        opp = normalize(row.get("eval_opponent_name")) or "unknown"
        by_opp[opp][1] += 1
        by_opp[opp][0] += res
        try:
            timeline.append((int(gid), res))
        except Exception:
            pass

    print("SUMMARY")
    if played:
        print(f"games={played}, wins={wins}, win_rate={wins/played:.4f}")
    else:
        print("No finished games with final_reward found in eval_samples_game_*.csv")

    print("\nBY OPPONENT")
    for opp, (w, p) in sorted(by_opp.items(), key=lambda x: -x[1][1]):
        if p:
            print(f"{opp}: {w}/{p} = {w/p:.3f}")

    timeline.sort(key=lambda x: x[0])
    W = max(1, int(args.window))
    roll: List[float] = []
    acc: List[int] = []
    for _, r in timeline:
        acc.append(r)
        if len(acc) > W:
            acc.pop(0)
        roll.append(sum(acc) / len(acc))
    print("\nROLLING WIN-RATE (last 10)")
    print(roll[-10:])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
