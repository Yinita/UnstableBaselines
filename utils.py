import os
import torch 
from typing import List, Dict

import ray 
from ray.util.placement_group import placement_group


def win_loss_reward_transformation(raw_rewards):
    if raw_rewards[0] > raw_rewards[1]:
        raw_rewards[0] = 1
        raw_rewards[1] = -1
    elif raw_rewards[0] < raw_rewards[1]:
        raw_rewards[0] = -1
        raw_rewards[1] = 1
    else:
        raw_rewards[0] = 0
        raw_rewards[1] = 0
    return raw_rewards


REWARD_TRANSFORMATIONS = {
    "win-loss": win_loss_reward_transformation,
    "raw": lambda raw_rewards: raw_rewards
}

def average_weights(weight_list: List[dict]) -> dict:
    avg_weights = {}
    n = len(weight_list)
    keys = weight_list[0].keys()

    for k in keys:
        stacked = sum(w[k] for w in weight_list)
        avg_weights[k] = stacked / n
    return avg_weights


def validate_requested_gpus(args):
    total_gpus = args.num_actors + args.num_learners
    available_gpus = torch.cuda.device_count()
    cpu_count = os.cpu_count()
    cpu_reserved = args.num_learners
    cpu_remaining = cpu_count - cpu_reserved
    safe_worker_count = int(cpu_count * 0.7)

    print("\n" + "="*48)
    print("System Resource Allocation Summary")
    print("="*48)

    def line(label, used, available, symbol=""):
        print(f"{label:<35} {used:>5} / {available:<5} {symbol}")

    line("GPUs for Actors", args.num_actors, available_gpus)
    line("GPUs for Learners", args.num_learners, available_gpus)
    line("Total GPUs Requested", total_gpus, available_gpus, "✅" if total_gpus == available_gpus else ("⚠️" if total_gpus < available_gpus else "❌"))

    print("-" * 48)
    line("Total CPUs", cpu_count, cpu_count)
    line("CPUs Reserved for Learners", cpu_reserved, cpu_count)
    line("CPUs Remaining for Workers", cpu_remaining, cpu_count)
    line("Safe Collector Limit (70%)", safe_worker_count, cpu_count)
    return total_gpus, safe_worker_count




def reserve_resources_for_learners(num_learners):
    # Reserve each learner 1 GPU and 1 CPU
    bundles = [{"CPU": 2, "GPU": 1} for _ in range(num_learners)]
    pg = placement_group(bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg
