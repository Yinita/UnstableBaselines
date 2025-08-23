#!/usr/bin/env python3
"""
Compute cumulative win-rate over evaluation games from eval CSVs.

Input rows (typical header):
  game_idx,turn_idx,pid,name,obs,full_action,extracted_action,step_info,final_reward,eval_model_pid,eval_opponent_name

Rule per game:
- For each game_idx, take the row where pid == eval_model_pid and read final_reward.
- win if final_reward > 0
- draw if final_reward == 0
- loss if final_reward < 0

We aggregate games in order and compute cumulative win-rate = wins / games.

Usage:
  python analysis/compute_cumwin_eval.py --base /abs/path/to/.../logs/samples --save_csv analysis/cumwin_eval_summary.csv --plot_path analysis/cumwin_eval.png
  python analysis/compute_cumwin_eval.py --files file1.csv --files file2.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

REQUIRED_COLS = {"game_idx", "pid", "final_reward", "eval_model_pid"}
OPTIONAL_COLS = {"eval_opponent_name"}


def normalize(x: object) -> str:
    return (str(x) if x is not None else "").strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cumulative win-rate from eval CSVs")
    ap.add_argument("--base", action="append", default=[], help="Directory containing eval CSVs (e.g., logs/samples)")
    ap.add_argument("--files", action="append", default=[], help="Specific CSV files; can repeat")
    ap.add_argument("--save_csv", default="", help="Optional output CSV path for per-game cumulative data")
    ap.add_argument("--plot_path", default="", help="Optional PNG path to save step-vs-cum_win_rate plot")
    return ap.parse_args()


def iter_eval_files(bases: List[str], files: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for f in files or []:
        af = os.path.abspath(f)
        if os.path.isfile(af) and af not in seen:
            seen.add(af)
            out.append(af)
    for b in bases or []:
        b = os.path.abspath(b)
        if not os.path.isdir(b):
            continue
        # be permissive: any CSV
        for p in glob.glob(os.path.join(b, "*.csv")):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(ap)
    out.sort()
    return out


def safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = normalize(x)
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def read_games_from_csv(path: str) -> Dict[str, float]:
    """Return map game_idx -> final_reward (for model pid only)."""
    result: Dict[str, float] = {}
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header: Optional[List[str]] = None
            for row in reader:
                if not row:
                    continue
                # Detect header rows dynamically (CSV may have repeated headers inside)
                if any(cell in (REQUIRED_COLS | OPTIONAL_COLS) for cell in row):
                    header = [normalize(h) for h in row]
                    continue
                if header is None:
                    # skip until we see a header
                    continue
                # map row
                if len(row) < len(header):
                    # malformed line
                    continue
                rec = {header[i]: row[i] for i in range(len(header)) if i < len(row)}
                # check required
                if not REQUIRED_COLS.issubset(rec.keys()):
                    continue
                pid = normalize(rec.get("pid"))
                eval_pid = normalize(rec.get("eval_model_pid"))
                if pid != eval_pid or eval_pid == "":
                    continue
                gidx = normalize(rec.get("game_idx"))
                fr = safe_float(rec.get("final_reward"))
                if gidx != "" and fr is not None:
                    # keep last occurrence per game_idx
                    result[gidx] = fr
    except Exception:
        # ignore file-level failure
        pass
    return result


def main() -> int:
    args = parse_args()
    files = iter_eval_files(args.base, args.files)
    if not files:
        print("No eval CSV files found. Provide --base /path/to/logs/samples or --files ...")
        return 1

    # Collect final_reward per game from all files
    game_rewards: List[Tuple[int, float]] = []  # (sort_key, reward)
    for fp in files:
        m = read_games_from_csv(fp)
        # try to sort by numeric game_idx when possible, else keep as is
        for g, r in m.items():
            try:
                k = int(g)
            except Exception:
                k = 10**12  # push non-numeric to end relatively
            game_rewards.append((k, r))

    if not game_rewards:
        print("No valid game records found in eval CSVs.")
        return 1

    game_rewards.sort(key=lambda x: x[0])

    # cumulative aggregation
    cum_wins = 0
    cum_games = 0
    rows: List[Tuple[int, float, int, int, float]] = []  # order, reward, cum_wins, cum_games, rate
    for order, reward in game_rewards:
        cum_games += 1
        if reward > 0:
            cum_wins += 1
        rate = cum_wins / cum_games
        rows.append((order, reward, cum_wins, cum_games, rate))

    # Print table
    print("order,final_reward,cum_wins,cum_games,cum_win_rate")
    for order, rr, cw, cg, rt in rows:
        print(f"{order},{rr},{cw},{cg},{rt:.6f}")
    print(f"\nSUMMARY: games={cum_games}, wins={cum_wins}, cum_win_rate={rows[-1][-1]:.4f}")

    # Save CSV
    if args.save_csv:
        outp = os.path.abspath(args.save_csv)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order", "final_reward", "cum_wins", "cum_games", "cum_win_rate"])
            for order, rr, cw, cg, rt in rows:
                w.writerow([order, rr, cw, cg, rt])
        print(f"Saved: {outp}")

    # Optional plot
    if args.plot_path:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            xs = [o for o, *_ in rows]
            ys = [rt for *_, rt in rows]
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker='o', linewidth=1.5)
            plt.xlabel('game order (by game_idx)')
            plt.ylabel('cumulative win-rate')
            plt.title('Cumulative Win-Rate (Eval)')
            plt.grid(True, alpha=0.3)
            outp = os.path.abspath(args.plot_path)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            plt.tight_layout()
            plt.savefig(outp, dpi=150)
            plt.close()
            print(f"Saved plot: {outp}")
        except Exception:
            print("[warn] matplotlib not available; skip plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
