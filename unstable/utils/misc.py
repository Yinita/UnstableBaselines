import csv, json, os
from typing import List, Optional
from unstable._types import GameInformation, Step, PlayerTrajectory


def write_training_data_to_file(batch, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward', "env_id", "step_info"])  # header
        for step in batch: writer.writerow([step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info])

def write_game_information_to_file(game_info: GameInformation, filename: str) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_idx", "turn_idx", "pid", "name", "obs", "full_action", "extracted_action", "step_info", "final_reward", "eval_model_pid", "eval_opponent_name"])
        writer.writeheader()
        for t in range(game_info.num_turns or len(game_info.obs)):
            pid = game_info.pid[t] if t < len(game_info.pid) else None
            row = {"game_idx": game_info.game_idx, "turn_idx": t, "pid": pid, "name": game_info.names.get(pid, ""), "obs": game_info.obs[t] if t < len(game_info.obs) else "", "full_action": game_info.full_actions[t] if t < len(game_info.full_actions) else "", "extracted_action": game_info.extracted_actions[t] if t < len(game_info.extracted_actions) else "", "step_info": json.dumps(game_info.step_infos[t] if t < len(game_info.step_infos) else {}, ensure_ascii=False), "final_reward": game_info.final_rewards.get(pid, ""), "eval_model_pid": game_info.eval_model_pid, "eval_opponent_name": game_info.eval_opponent_name}
            writer.writerow(row)

def write_eval_data_to_file(batch: List[Step], filename: str):
    """Save evaluation data to a CSV file.
    
    Args:
        batch: List of Step objects containing evaluation data
        filename: Path to the output CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['pid', 'obs', 'act', 'reward', 'env_id', 'step_info'])
        
        # Write each step
        for step in batch:
            writer.writerow([
                step.pid, 
                step.obs, 
                step.act, 
                step.reward, 
                step.env_id, 
                json.dumps(step.step_info, ensure_ascii=False) if step.step_info else '{}'
            ])


def write_samples_to_file(trajs: List[PlayerTrajectory], filename: str, env_id: Optional[str] = None):
    """Write collected samples (trajectories) to a CSV file.
    Each row corresponds to one step within a PlayerTrajectory.

    Columns: pid, obs, act, extracted_act, logp, env_id, step_info
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pid", "obs", "act", "extracted_act", "logp", "env_id", "step_info"])  # header

        # counters and previews
        total_rows = 0
        preview_rows = []  # store up to 2 preview tuples

        for traj_idx, traj in enumerate(trajs):
            # Use max between actions and extracted_actions to avoid dropping rows when raw action is missing
            steps_len = min(len(traj.obs), max(len(traj.actions), len(traj.extracted_actions)))
            # per-trajectory lens debug
            try:
                print(
                    f"[write_samples_to_file] traj#{traj_idx} pid={traj.pid} lens: "
                    f"obs={len(traj.obs)} acts={len(traj.actions)} extr={len(traj.extracted_actions)} "
                    f"logps={len(traj.logps)} step_infos={len(traj.step_infos)} -> steps_len={steps_len}"
                )
            except Exception:
                pass

            if steps_len == 0:
                # light-weight warning without introducing logger dependency
                try:
                    print(f"[write_samples_to_file] Warning: zero steps for pid={traj.pid}; lens obs={len(traj.obs)}, acts={len(traj.actions)}, extracted={len(traj.extracted_actions)}")
                except Exception:
                    pass

            for i in range(steps_len):
                pid = traj.pid
                obs = traj.obs[i]
                act = traj.actions[i] if i < len(traj.actions) else (traj.extracted_actions[i] if i < len(traj.extracted_actions) else "")
                extracted = traj.extracted_actions[i] if i < len(traj.extracted_actions) else (traj.actions[i] if i < len(traj.actions) else "")
                logp = traj.logps[i] if i < len(traj.logps) else 0.0
                step_info = traj.step_infos[i] if i < len(traj.step_infos) else {}
                writer.writerow([
                    pid,
                    obs,
                    act,
                    extracted,
                    logp,
                    env_id if env_id is not None else (step_info.get("env_id") if isinstance(step_info, dict) else None),
                    json.dumps(step_info, ensure_ascii=False) if step_info else '{}'
                ])
                total_rows += 1

                # capture preview of first 2 rows across all trajs
                if len(preview_rows) < 2:
                    try:
                        def _trunc(s, n=160):
                            try:
                                return (s[:n] + '…') if isinstance(s, str) and len(s) > n else s
                            except Exception:
                                return s
                        preview_rows.append({
                            "pid": pid,
                            "obs": _trunc(obs, 160),
                            "act": _trunc(act, 160),
                            "extracted": _trunc(extracted, 160),
                            "logp": logp
                        })
                    except Exception:
                        pass

        # file-level summary
        try:
            print(f"[write_samples_to_file] Wrote {total_rows} rows to {os.path.abspath(filename)} (env_id={env_id})")
            if preview_rows:
                print(f"[write_samples_to_file] Preview (up to 2 rows): {json.dumps(preview_rows, ensure_ascii=False)}")
        except Exception:
            pass
