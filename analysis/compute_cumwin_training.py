#!/usr/bin/env python3
"""
Compute cumulative win-rate over training steps from training_data/*.csv.

Input CSV schema per row: pid,obs,act,reward,env_id,step_info
Rule: For each step file (e.g., train_data_step_0.csv), use the last step_info
value in the file, parse it as a dict and take raw_reward:
- win   if raw_reward > 0
- draw  if raw_reward == 0
- loss  if raw_reward < 0

We then compute cumulative win-rate across steps (wins/steps).

Usage:
  python analysis/compute_cumwin_training.py --base /abs/path/to/.../training_data
  python analysis/compute_cumwin_training.py --files /abs/path/train_data_step_0.csv --files ...

Outputs a small table to stdout.
"""
from __future__ import annotations
import argparse
import ast
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

STEP_RE = re.compile(r"train_data_step_(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cumulative win-rate over training steps from training_data/*.csv")
    ap.add_argument("--base", default="", help="Directory containing train_data_step_*.csv")
    ap.add_argument("--files", action="append", default=[], help="Explicit CSV file(s). Can repeat.")
    ap.add_argument("--save_csv", default="", help="Optional path to save the per-step summary CSV.")
    ap.add_argument("--plot_path", default="", help="Optional path to save PNG plot of step vs cum_win_rate.")
    return ap.parse_args()


def find_step_files(base: str, files: List[str]) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    if files:
        for p in files:
            m = STEP_RE.search(os.path.basename(p))
            if not m:
                continue
            step = int(m.group(1))
            candidates.append((step, os.path.abspath(p)))
    if base:
        base = os.path.abspath(base)
        for p in glob.glob(os.path.join(base, "train_data_step_*.csv")):
            m = STEP_RE.search(os.path.basename(p))
            if not m:
                continue
            step = int(m.group(1))
            candidates.append((step, os.path.abspath(p)))
    # unique by path, keep lowest step association if duplicates
    seen: Dict[str, int] = {}
    for step, path in candidates:
        if path not in seen or step < seen[path]:
            seen[path] = step
    pairs = [(st, pa) for pa, st in seen.items()]
    pairs.sort(key=lambda x: x[0])
    return pairs


def extract_last_step_info_reward(csv_path: str) -> Optional[float]:
    last_info: Optional[str] = None
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            # normalize headers
            reader.fieldnames = [h.strip() if h else h for h in reader.fieldnames]
            for row in reader:
                info = row.get("step_info")
                if info is not None and str(info).strip() != "":
                    last_info = str(info).strip()
    except Exception:
        return None
    if not last_info:
        return None
    # Parse python-literal-like dict (single quotes). Use ast.literal_eval safely.
    try:
        data = ast.literal_eval(last_info)
        if isinstance(data, dict) and "raw_reward" in data:
            val = data["raw_reward"]
            try:
                return float(val)
            except Exception:
                return None
    except Exception:
        return None
    return None


def main() -> int:
    args = parse_args()
    pairs = find_step_files(args.base, args.files)
    if not pairs:
        print("No training step CSVs found. Provide --base /path/to/training_data or --files ...")
        return 1

    cum_wins = 0
    cum_games = 0
    rows: List[Tuple[int, float, int, int, float]] = []  # step, raw_reward, cum_wins, cum_games, cum_win_rate

    for step, path in pairs:
        rr = extract_last_step_info_reward(path)
        # classify
        win = 1 if (rr is not None and rr > 0) else 0 if (rr is not None and rr == 0) else 0
        # Only wins accumulate in numerator; denominator grows every step
        cum_games += 1
        cum_wins += 1 if (rr is not None and rr > 0) else 0
        rate = cum_wins / cum_games if cum_games else 0.0
        rows.append((step, rr if rr is not None else float('nan'), cum_wins, cum_games, rate))

    # Print table
    print("step,raw_reward,cum_wins,cum_games,cum_win_rate")
    for step, rr, cw, cg, rt in rows:
        print(f"{step},{rr},{cw},{cg},{rt:.6f}")
    if rows:
        last_step, last_rr, last_cw, last_cg, last_rt = rows[-1]
        print(f"\nSUMMARY: steps={last_cg}, wins={last_cw}, cum_win_rate={last_rt:.4f}")

    if args.save_csv:
        outp = os.path.abspath(args.save_csv)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "raw_reward", "cum_wins", "cum_games", "cum_win_rate"])
            for step, rr, cw, cg, rt in rows:
                w.writerow([step, rr, cw, cg, rt])
        print(f"Saved: {outp}")

    # Optional plot
    if args.plot_path:
        if plt is None:
            print("[warn] matplotlib is not available; cannot save plot.")
        else:
            steps = [s for s, *_ in rows]
            rates = [rt for *_, rt in rows]
            plt.figure(figsize=(8,4))
            plt.plot(steps, rates, marker='o', linewidth=1.5)
            plt.xlabel('step')
            plt.ylabel('cumulative win-rate')
            plt.title('Cumulative Win-Rate over Training Steps')
            plt.grid(True, alpha=0.3)
            outp = os.path.abspath(args.plot_path)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            plt.tight_layout()
            plt.savefig(outp, dpi=150)
            plt.close()
            print(f"Saved plot: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
