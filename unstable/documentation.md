# Unstable Baselines  – Documentation

> **Version:** 0.1  ·  **Last updated:** 2025‑06‑21

The documentation is currently written by GPT. We will update this (hopefully) soon.

---

## Table of Contents

1. [What Is Unstable Baselines?](#what-is-unstable-baselines)
2. [System Overview](#system-overview)
3. [Quick‑Start Guide](#quick‑start-guide)
4. [Core Runtime Components](#core-runtime-components)
5. [Reward Transformation Pipeline](#reward-transformation-pipeline)
6. [Algorithms](#algorithms)
7. [Configuration Reference](#configuration-reference)
8. [Extending the Framework](#extending-the-framework)
9. [File‑to‑Module Map](#file‑to‑module-map)
10. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 1 · What Is Unstable Baselines?

**Unstable Baselines** is a lightweight reinforcement‑learning research harness focused on *self‑play* for text‑based games. It couples:

* **Ray** – easy, elastic distributed execution.
* **vLLM** – high‑throughput inference with LoRA hot‑swapping.
* **TextArena** – a growing suite of competitive text games.

The goal is to iterate **quickly** on small language models (< 8B params) and benchmark new ideas in *reasoning and agentic behaviour*.

---

## 2 · System Overview  <a id="system-overview"></a>

The runtime can be thought of as three asynchronous loops:

```
    ┌───────────────┐                           ┌───────────────┐                           ┌───────────────┐
    │               │    Register new lora      │               │        Get Loss &         │               │
    │   Model Pool  │◀──────────────────────────│    Learner    │◀─────────────────────────▶│   Algorithm   │
    │               │       checkpoint          │               │      update weights       │               │
    └───────────────┘                           └───────────────┘                           └───────────────┘ 
           ▲ │                                       ▲     │ 
           │ │ Sample                      If enough │     │ Check if enough
    Update │ │ Opponent                   data, pull │     │ data for training
 Trueskill │ │                        the next batch │     │ is available
           │ ▼                                       │     ▼
    ┌───────────────┐                          ┌───────────────┐                      
    │               │     Process and store    │               │                      
    │   Collector   │─────────────────────────▶│   StepBuffer  │                      
    │               │  collected Trajectories  │               │                      
    └───────────────┘                          └───────────────┘                      
           ▲ │                      
           │ │ Maintain     
    return │ │ Pool of 
Trajectory │ │ n parallel      
           │ │ workers   
           │ ▼
     ┌─────────────┐
     │  run_game() │
     │  train\eval │
     └─────────────┘
```



* **Collector** instances roll games with the latest learner checkpoint vs. opponents sampled by the **Model Pool**.
* End‑of‑game rewards & formatted trajectories land in the **Step Buffer**.
* The **Learner** periodically drains a batch, performs a gradient step, saves a LoRA checkpoint and registers it with the **Model Pool**.
* The **Tracker** aggregates metrics; the **Terminal Interface** turns them into a live Rich dashboard.

---

## 3 · Quick‑Start Guide  <a id="quick‑start-guide"></a>

```bash
# Prerequisites
conda create -n unstable python=3.12 && conda activate unstable
pip install --upgrade ray torch vllm textarena wandb rich trueskill peft transformers pynvml

# Clone repository
$ git clone https://github.com/<you>/unstable-baselines && cd unstable-baselines
```

### Minimal training script

```python
import time, ray, unstable
import unstable.reward_transformations as retra

ray.init(namespace="unstable")
tracker = unstable.Tracker.options(name="Tracker").remote(run_name="demo", wandb_project="UB")
step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
    max_buffer_size=768,
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.WinDrawLossFormatter()]),
    step_reward_transformation=None,
    sampling_reward_transformation=None,
)
model_pool = unstable.ModelPool.options(name="ModelPool").remote(sample_mode="mirror", max_active_lora=3, tracker=tracker)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration=-1))
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=2,
    step_buffer=step_buffer,
    model_pool=model_pool,
    tracker=tracker,
    vllm_config={
        "model_name": "Qwen/Qwen3-1.7B-base",
        "max_parallel_seq": 64,
        "max_tokens": 2048,
        "max_loras": 4,
        "lora_config": {"lora_rank": 32},
        "max_model_len": 8192
    },
    training_envs=[("Nim-v0-train", 2, "qwen3-zs")],
    evaluation_envs=[],
)
learner = unstable.StandardLearner.options(num_gpus=1, name="Learner").remote(
    model_name="Qwen/Qwen3-1.7B-base",
    step_buffer=step_buffer,
    model_pool=model_pool,
    tracker=tracker,
    algorithm=unstable.algorithms.Reinforce(),
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    lora_cfg={"lora_rank": 32},
)
collector.collect.remote(num_workers=256, num_eval_workers=0)
ray.get(learner.train.remote(100))
```

> **Tip:** run `python3 -m unstable.terminal_interface` in a separate terminal to launch the live dashboard.

---

## 4 · Core Runtime Components  <a id="core-runtime-components"></a>

### 4.1 `VLLMActor`  <a id="41-vllmactor"></a>

High‑throughput inference wrapper around **vLLM** with on‑the‑fly LoRA switching. Key points:

* Uses `EngineArgs(enable_lora=True)` so LoRA adapters are loaded only when referenced.
* Maintains:

  * per‑request token counters for accurate *tok/s* reporting.
  * async batcher that merges queued prompts every 20 ms.
* Periodically pushes stats through `tracker.log_inference()`.

### 4.2 Model Pool  <a id="42-model-pool"></a>

Manages *which* opponent checkpoint to sample for self‑play.

| Concept                | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| **`add_checkpoint()`** | Registers a new learner LoRA, inherits `μ, σ` from previous ckpt.        |
| **Sampling modes**     | `mirror`, `lagged`, `random`, `match-quality`, `ts-dist`, `exploration`. |
| **Active pool**        | Keeps ≤ `max_active_lora` checkpoints *active* to bound GPU memory.      |
| **TrueSkill**          | Uses `trueskill.TrueSkill(beta=4.0)` for rating updates.                 |

### 4.3 Collector  <a id="43-collector"></a>

Launches episodes via `play_episode.remote`.

* *Train jobs* pit the latest ckpt against a sampled opponent.
* *Eval jobs* pit the latest ckpt against a fixed reference model.
* Results funnel back into the Step Buffer + Tracker.

### 4.4 Step Buffer  <a id="44-step-buffer"></a>

A bounded FIFO of `Step` objects ready for the learner. Features:

* Reward transforms executed **in‑place** when a trajectory is added.
* Optional normalisation **per env** during sampling.
* CSV dump of every sampled batch for offline analysis.

### 4.5 StandardLearner  <a id="45-standardlearner"></a>

Single‑process learner that:

* Builds a PEFT/LoRA model, optionally cold‑starting from a LoRA path.
* Runs gradient checkpointing + Flash‑Attention 2.
* Minibatch PPO‑style updates via a pluggable `BaseAlgo` (default: REINFORCE).
* Auto‑saves LoRA adapters every `save_every` steps and registers them with the Model Pool.

### 4.6 Tracker  <a id="46-tracker"></a>

Central metric logger. All remote actors call it.

* Aggregates rolling windows → pushes to **Weights & Biases**.
* Exposes `get_interface_info()` for the dashboard.

### 4.7 Terminal Interface  <a id="47-terminal-interface"></a>

Rich‑TUI monitoring showing GPU usage, TrueSkill ladder, buffer stats and match heatmap. Run stand‑alone via:

```bash
python -m unstable.terminal_interface
```

---

## 5 · Reward Transformation Pipeline  <a id="reward-transformation-pipeline"></a>

The framework separates **when** a reward is applied:

1. **Final**   (`ComposeFinalRewardTransforms`) – operates on the *game‑level* reward vector.
2. **Step**    (`ComposeStepRewardTransforms`) – per‑step shaping **after** final reward is known.
3. **Sampling**(`ComposeSamplingRewardTransforms`) – modifies rewards just before a batch is returned.

All three stages are optional and fully composable.

---

## 6 · Algorithms  <a id="algorithms"></a>

New RL algorithms inherit from `BaseAlgo`:

```python
class MyAlgo(BaseAlgo):
    def prepare_batch(self, steps):  ...  # encode obs + actions → tensors
    def update(self, batch, scaling): ...  # compute loss, back‑prop – *do not* .step()
```

Current built‑ins:

* **REINFORCE** – vanilla policy‑gradient, cross‑entropy on entire trajectory.

---

## 7 · Configuration Reference  <a id="configuration-reference"></a>

| Symbol             | Where         | Meaning                     | Default  |
| ------------------ | ------------- | --------------------------- | -------- |
| `SAMPLE_MODE`      | user script   | Opponent sampling strategy  | `mirror` |
| `MAX_PARALLEL_SEQ` | `vllm_config` | Max queued prompts per GPU  |  128     |
| `BATCH_SIZE`       | Learner       | Steps per learner update    | 384      |
| `GRAD_CLIP`        | Learner       | `clip_grad_norm_` threshold |  0.2     |
| `BUFFER_SIZE`      | StepBuffer    | Max stored steps            | 768      |

Refer to individual class constructors for the full list.

---

## 8 · Extending the Framework  <a id="extending-the-framework"></a>

### Add a New Game Environment

1. Implement the environment in **TextArena**.
2. Add an entry to `TRAINING_ENVS` / `EVALUATION_ENVS`.

### Plug a Custom Reward Transform

```python
class ShapedWinBonus(retra.FinalRewardTransform):
    def __call__(self, x, env_id=None):
        return {pid: r + 0.1 if r > 0 else r for pid, r in x.items()}

step_buffer = ub.StepBuffer.options(...).remote(
    final_reward_transformation=retra.ComposeFinalRewardTransforms([ShapedWinBonus()])
)
```

### Write a New Algorithm

Just subclass `BaseAlgo` and pass an instance to `StandardLearner`.

---

## 9 · File‑to‑Module Map  <a id="file‑to‑module-map"></a>

| File                           | Primary Classes / Functions | Purpose                           |
| ------------------------------ | --------------------------- | --------------------------------- |
| `actor.py`                     | `VLLMActor`                 | Batched inference + LoRA hot‑swap |
| `collector.py`                 | `Collector`, `play_episode` | Rollouts & evaluation             |
| `model_pool.py`                | `ModelPool`                 | Opponent management, TrueSkill    |
| `buffer.py`                    | `StepBuffer`                | Sample buffer + reward transforms |
| `learners/standard_learner.py` | `StandardLearner`           | Training loop                     |
| `core.py`                      | `Trajectory`, `Step`, etc.  | Shared dataclasses                |
| `trackers.py`                  | `Tracker`                   | Metrics / logging                 |
| `terminal_interface.py`        | `TerminalInterface`         | Live dashboard                    |
| `algorithms/reinforce.py`      | `Reinforce`                 | Default policy‑gradient           |

---

## TODO
- add a table of recommended settings for specific hardware set-ups (i.e. whether activation checkpointing is necessary, what seq-len to use etc.)



If you have any issues at all, please feel free to e-mail me: guertlerlo@cfar.a-star.edu.sg