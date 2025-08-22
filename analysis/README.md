# Analysis Utilities

This folder contains scripts to analyze run outputs under `outputs/.../logs/samples/`.

## Win-rate computation

Script: `analysis/compute_winrate.py`

It scans one or more `logs/samples` directories and aggregates win-rate from `eval_samples_game_*.csv` files.

Key points:
- Robust to mixed/duplicated headers inside a CSV.
- Uses `final_reward` from rows where `pid == eval_model_pid` per `game_idx`.
- Reports overall win-rate, per-opponent breakdown (`eval_opponent_name`), and a rolling win-rate timeline.

### Usage

- Latest run auto-detection (finds most recently modified `.../logs/samples`):

```
python analysis/compute_winrate.py
```

- Explicit directory:

```
python analysis/compute_winrate.py --base /abs/path/to/outputs/.../logs/samples
```

- Multiple runs with glob:

```
python analysis/compute_winrate.py --glob "/abs/outputs/**/logs/samples"
```

- Control rolling window size (default 100 games):

```
python analysis/compute_winrate.py --base /abs/path/.../logs/samples --window 200
```

If you want further breakdowns (e.g., by role/team), let us know the relevant column names and we will extend the script.
