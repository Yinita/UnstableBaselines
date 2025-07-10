# Unstable Baselines Documentation

> **Version:** 0.2 · **Last Updated:** 2025-07-10
>
> This release introduces a dedicated **sampler** layer (`samplers/`), a composable **runtime** builder (`runtime.build()`), and separate **REINFORCE** / **A2C** learners.  `ModelPool` has been renamed **ModelRegistry**, and the data layer now exposes both **StepBuffer** *and* **EpisodeBuffer**.

---

code‑line counts
`v0.1.0`  1 144
`v0.2.0`  1 267 (TODO confirm and plot final)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)

   * Installation
   * Quick Start
3. [Architecture Overview](#architecture-overview)
4. [Core Modules](#core-modules)
5. [Reward Transformations](#reward-transformations)
6. [Algorithms](#algorithms)
7. [Configuration Reference](#configuration-reference)

---

## Introduction <a id="introduction"></a>

The key piece holding **Unstable Baselines** together is the **Collector**.  It maintains a pool of `num_train_workers` and `num_eval_workers` games in flight.  A **GameScheduler** decides *what* to run next by querying the **EnvSampler** (which environment?) and **ModelSampler** (which opponent?).  When a game finishes, trajectories stream into a replay **Buffer**, metrics go to the **Tracker**, and TrueSkill updates flow back to the **ModelRegistry**.

---

## Getting Started <a id="getting-started"></a>

### Installation

```bash
pip install unstable-rl
```

### Quick Start

Below we reproduce the *classic* quick‑start snippet in the same style as earlier docs – numbered comments explain each line.

```python
# 1) import the high‑level facade
import unstable

# 2) declare the training & evaluation environments
train_envs = [
    unstable.TrainEnvSpec(
        env_id="SimpleTak",       # game name
        num_players=2,             # 2‑player zero‑sum
        num_actors=1,              # 1 learner actor, rest are opponents
        prompt_template="qwen3-zs" # prompt formatting key
    )
]

# fixed baseline for eval
GEMINI = "google/gemini-2.0-flash-lite-001"

eval_envs = [
    unstable.EvalEnvSpec(env_id="SimpleTak", num_players=2, prompt_template="qwen3-zs"),
    unstable.EvalEnvSpec(env_id="KuhnPoker", num_players=2, prompt_template="qwen3-zs")
]

# 3) spawn every actor via the runtime builder
run = unstable.runtime.build(
    model_name="Qwen/Qwen3-1.7B-Base",   # HF base model
    train_envs=train_envs,
    eval_envs=eval_envs,
    algorithm="reinforce",              # learner algo
    iterations=200,                      # learner update steps
    opponent_fixed=[GEMINI],             # baseline list
    num_train_workers=384,               # concurrent self‑play games
    num_eval_workers=16,                 # concurrent eval games
)

# 4) asynchronous execution helpers
run.start()  # launches Collector + Learner
# run.wait()   # block until Learner finishes (200 updates)
run.stop()   # tear down actors & Ray runtime
```

In a nutshell, the **Collector** will keep `384` self‑play and `16` evaluation games running in parallel.  The **Learner** polls the **StepBuffer**; once ≥ `batch_size` steps are available it performs one gradient step, saves a new LoRA adapter, and registers it with the **ModelRegistry**.

---

## Architecture Overview <a id="architecture-overview"></a>

The original ASCII diagram is preserved below – only the label *Model Pool* → *Model Registry* changed.

```
 ┌─────────┐ ┌─────────┐             ┌────────────┐
 │   Env   │ │  Model  │ Get Models  │    Model   │
 │ Sampler │ │ Sampler │◀─────────── │  Registry  │
 └─────────┘ └─────────┘             └────────────┘ 
      │          │                         ▲
      │Sample    │Sample                   │Push
      │Env       │Opponent                 │Checkpoint 
      ▼          ▼                         │
    ┌───────────────┐              ┌───────────────┐
    │               │              │               │
    │ GameScheduler │              │    Learner    │
    │               │              │               │
    └───────────────┘              └───────────────┘
           ▲ │                            ▲ │ 
           │ │ Sample           If enough │ │ Check if enough
    Update │ │ GameSpec        data, pull │ │ data for training
           │ │             the next batch │ │ is available
           │ ▼                            │ ▼
    ┌───────────────┐               ┌────────────┐
    │               │      Send     │            │
    │   Collector   │──────────────▶│   Buffer   │
    │               │ Trajectories  │            │
    └───────────────┘               └────────────┘
           ▲ │
           │ │ Maintain
    return │ │ Pool of 
Trajectory │ │ n parallel
           │ │ workers
           │ ▼
     ┌─────────────┐
     │  run_game() │
     │  train/eval │
     └─────────────┘
```

---

## Core Modules <a id="core-modules"></a>

| Module                | Source File                  | One‑line Purpose                                  |
| --------------------- | ---------------------------- | ------------------------------------------------- |
| **VLLMActor**         | `actor.py`                   | GPU‑bound async text generation + LoRA hot‑swap   |
| **Collector**         | `collector.py`               | Orchestrates episode rollout & trajectory capture |
| **ModelRegistry**     | `model_registry.py`          | Keeps checkpoints & TrueSkill ratings             |
| **EnvSampler**        | `samplers/env_samplers.py`   | Uniform‑random or reward‑aware env selection      |
| **ModelSampler**      | `samplers/model_samplers.py` | Self‑play / fixed opponent / lagged sampling      |
| **GameScheduler**     | `game_scheduler.py`          | Converts sampler outputs into GameSpecs           |
| **StepBuffer**        | `buffers.py`                 | Stores Steps for on‑policy learners               |
| **EpisodeBuffer**     | `buffers.py`                 | Stores full episodes for off‑policy algorithms    |
| **Learners**          | `learners/`                  | REINFORCE & A2C LoRA fine‑tuning                  |
| **Tracker**           | `trackers.py`                | Centralised metrics & W\&B logging                |
| **TerminalInterface** | `terminal_interface.py`      | Live Rich dashboard                               |
| **runtime.build()**   | `runtime.py`                 | High‑level factory that wires everything together |


---

### Component Reference

<details>
<summary><strong>VLLMActor (<code>actor.py</code>)</strong><a id="actor"></a></summary>

GPU‑bound, async wrapper around a single **vLLM** engine.  Queues generation requests, batches them every 20 ms, supports on‑the‑fly **LoRA** hot‑swap, and reports queue size + tokens‑per‑second back to the **Tracker**.

* **Public API**  – `submit_prompt(prompt, lora_path=None) -> str`
* **Background Tasks** – `_batch_loop()` for stepping the engine, `_report_loop()` for metrics.
* **LoRA pool** – maps adapter paths to numeric IDs so vLLM can switch weights without re‑loading the base model.

</details>

<details>
<summary><strong>Collector (<code>collector.py</code>)</strong><a id="collector"></a></summary>

Orchestrates episode rollout.  Maintains a round‑robin iterator over a pool of **VLLMActor** GPUs, spawns remote `run_game()` tasks, and tracks them in `flight` with accompanying `TaskMeta`.

* Submits *training* games until `<num_train_workers>` are running and *evaluation* games until `<num_eval_workers>` are running.
* Handles results via `_post_train()` (streams trajectories to **Buffer**, updates **ModelRegistry**) or `_post_eval()` (logs rewards).
* Back‑pressure: pauses submission when `Buffer.continue_collection()` returns `False`.

</details>

<details>
<summary><strong>ModelRegistry (<code>model_registry.py</code>)</strong><a id="model-registry"></a></summary>

Central store of <em>all</em> opponents – learner checkpoints and fixed baselines.  Uses **TrueSkill** to track skill, records pair‑wise match counts, and exposes two key calls:

* `add_checkpoint(uid, path, iteration)` – inherit μ/σ from previous ckpt.
* `update_ratings(uids, scores, env_id)` – batch TrueSkill update after every game.

`get_current_ckpt()` returns the latest learner UID; `sample()` is delegated to a **ModelSampler** strategy.

</details>

<details>
<summary><strong>EnvSampler / ModelSampler (<code>samplers/*.py</code>)</strong><a id="samplers"></a></summary>

* **EnvSampler** – currently `UniformRandomEnvSampler`; plug‑in point for curriculum or reward‑aware scheduling.
* **ModelSampler** – decides which opponent UID to fight.  Built‑in `FixedOpponentModelSampler` uniformly draws from registered baselines, but you can subclass to implement lagged or TrueSkill‑match‑quality sampling.

Both are pure Python – no Ray actors – so they stay cheap and composable.

</details>

<details>
<summary><strong>GameScheduler (<code>game_scheduler.py</code>)</strong><a id="game-scheduler"></a></summary>

Small Ray actor that fuses `EnvSampler` + `ModelSampler` into concrete **GameSpec** objects.  Keeps an internal `_game_idx` counter (seed) and a `_running_jobs` dict so it can compute average actor/opponent reward when a game finishes.

</details>

<details>
<summary><strong>StepBuffer / EpisodeBuffer (<code>buffers.py</code>)</strong><a id="buffer"></a></summary>

Replay store living on a Ray actor.  Two flavours:

* **StepBuffer** – flattens trajectories into `Step` objects; ideal for on‑policy REINFORCE style updates.
* **EpisodeBuffer** – keeps whole episodes; useful for value‑based or off‑policy algorithms.

Both support reward‑shaping via three hook pipelines (final / per‑step / sampling‑time) and evict random samples once `len(buffer) > max_buffer_size`.

</details>

<details>
<summary><strong>REINFORCELearner (<code>learners/reinforce_learner.py</code>)</strong><a id="reinforce"></a></summary>

Pure policy‑gradient learner.  Pulls <code>batch\_size</code> steps, splits into <code>mini\_batch\_size</code> chunks for gradient accumulation, computes <code>-advantage × log p</code>, clips gradients, steps AdamW, saves a LoRA adapter every learner update and registers it with **ModelRegistry**.

</details>

<details>
<summary><strong>A2CLearner (<code>learners/a2c_learner.py</code>)</strong><a id="a2c"></a></summary>

Actor‑critic sibling.  Builds an extra LoRA‑wrapped **critic** head via `learners.utils.build_peft_model()` and learns both policy & value in tandem.  Uses GAE for advantage computation; optional reward normalisation with `NormalizeRewardsByEnv`.

</details>

<details>
<summary><strong>Tracker (<code>trackers.py</code>)</strong><a id="tracker"></a></summary>

Lightweight Ray actor that buffers scalar metrics in memory, aggregates them into moving means, and flushes to **Weights & Biases** every 64 s (optional).  Also exposes `get_interface_info()` consumed by the live **TerminalInterface**.

</details>

<details>
<summary><strong>TerminalInterface (<code>terminal_interface.py</code>)</strong><a id="terminal-interface"></a></summary>

Rich‑based curses UI that renders:

* **GPU panel** – live tokens/sec, power draw, memory usage.
* **Collection stats** – format success, invalid move rate, game length.
* **TrueSkill bar chart** – μ/σ for every checkpoint and baseline.
* **Heat‑map** – match counts between top‑N models.

It refreshes every 2 s and resizes gracefully.

</details>

<details>
<summary><strong>Runtime Builder (<code>runtime.py</code>)</strong><a id="runtime"></a></summary>

One‑stop factory that spins up **Tracker**, **ModelRegistry**, **Buffer**, **GameScheduler**, **Collector**, and **Learner** actors; wires them together, and returns a handle with `start() / wait() / stop()` helpers.  Pass `algorithm="reinforce"` or `"a2c"` to pick the learner class.

</details>

---

---
## Reward Transformations <a id="reward-transformations"></a>

Reward shaping in **Unstable Baselines** is fully modular—three independent pipelines can be stacked to turn sparse game outcomes into dense learning signals.

| Pipeline            | Runs                                  | Typical Use-Case                                                                      |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| **Final Reward**    | *once* per game                       | Balance first-player advantage, convert raw env scores into `{-1, 0, 1}` etc.         |
| **Step Reward**     | every step                            | Give tiny bonuses for valid format, penalise invalid moves, distance-to-goal shaping. |
| **Sampling Reward** | right before a learner batch is drawn | Normalise or clip advantages, on-policy GAE style transforms.                         |

### Base Interfaces

```python
class FinalRewardTransform:
    def __call__(self, reward: float, pid: int, env_id: str|None=None) -> float: ...

class StepRewardTransform:
    def __call__(self, player_traj, step_index: int, reward: float) -> float: ...

class SamplingRewardTransform:
    def __call__(self, steps: list[Step]) -> list[Step]: ...
```

Compose multiple transforms via the provided helpers:

```python
final_t  = retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()])
step_t   = retra.ComposeStepRewardTransforms([
    retra.RewardForFormat(reward=0.25, penalty=0.0),
    retra.PenaltyForInvalidMove(penalty=-1.0)
])
sample_t = retra.ComposeSamplingRewardTransforms([
    retra.NormalizeRewardsByEnv(z_score=True)
])
```

Pass these into `StepBuffer` / `EpisodeBuffer` at construction time (the *runtime* builder exposes keyword hooks).

#### Built-in Final Reward Transforms

| Class                         | Effect                                                       |
| ----------------------------- | ------------------------------------------------------------ |
| `RoleAdvantageFormatter`      | Subtract a global EMA of each player-ID’s historical reward. |
| `RoleAdvantageByEnvFormatter` | Same, but tracked per-environment.                           |

#### Built-in Step Reward Transforms

| Class                                    | Effect                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `RewardForFormat(reward, penalty)`       | Adds *reward* when the agent encloses its answer in `\boxed{}` and *penalty* otherwise. |
| `PenaltyForInvalidMove(reward, penalty)` | Adds *penalty* if the env marks the step as invalid.                                    |

#### Built-in Sampling Reward Transforms

| Class                                  | Effect                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| `NormalizeRewards(z_score=False)`      | Mean-centres (and optionally z-scores) rewards in the batch. |
| `NormalizeRewardsByEnv(z_score=False)` | Same but computed separately per env-ID bucket.              |

Add your own transform by subclassing the relevant base class and appending it to the compose helper.

---

## Algorithms <a id="algorithms"></a>

### REINFORCE (On-Policy)

Minimal policy-gradient on the frozen backbone + LoRA head.

* **Advantage** = per-step reward (already shaped) – no baseline by default.
* **Loss** = `-log π(a|s) × advantage` averaged over sequence tokens.
* **Token masking** – prompt tokens are masked out so only generated tokens contribute to the loss.
* **Truncation** – `max_train_len` limits context seen by the loss while `max_generation_len` limits new tokens produced during rollouts.
* **Gradient Accumulation** – `batch_size // mini_batch_size` forward/backward passes before one `optimizer.step()`.



### A2C (Actor-Critic)

Adds a separate **critic** value head (LoRA wrapped) on the same backbone.

1. **Rollout** as usual, but learner periodically runs the critic to produce state-values for every step.
2. **GAE** computes advantages + returns.
3. **Policy Loss** identical to REINFORCE (but uses GAE advantage).
4. **Value Loss** = 0.5 × MSE(return, value\_pred).
5. Joint optimisation with two AdamW optimisers (policy & critic).

Key config:

```python
learner.initialize_algorithm(
    infer_mini_batch_size=16,     # critic forward batch-size
    critic_learning_rate=1e-5,
    normalize_adv=True,           # optional SamplingRewardTransform
)
```

### Adding Your Own

Create `learners/my_algo_learner.py`, subclass `BaseLearner`, implement:

* `initialize_algorithm(...)`
* `_update(batch)` – must `backward()` on model parameters and return a metrics dict.
  Expose it from `learners/__init__.py` and pass `algorithm="my_algo"` to `runtime.build()`.

\--- <a id="algorithms"></a>

* **REINFORCE** – minimal policy‑gradient with advantage shaping.
* **A2C** – actor‑critic with GAE; critic shares the frozen backbone + LoRA value head.
* *Extend your own* – derive from `BaseLearner` and plug into `runtime.build()`.

---
## Configuration Reference <a id="configuration-reference"></a>

Below are recommended configurations for different VRAM capacities. We strongly recommend using **one GPU as a learner** and dedicating **remaining GPUs as actors** for inference.
Currently you will need at least 2 gpus to run the code (1 learner and 1 actor); we plan to relax this requirement in the future. Here are some rough guidelines on which parameter settings to use for what model size and hardware:

### Qwen3 1.7B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 12GB | TODO                     | TODO                   | TOOD                    |
| 16GB | TOOD                     | TOOD                   | TOOD                    |
| 24GB | TOOD                     | TOOD                   | TOOD                    |
| 48GB+| TOOD                     | TOOD                   | TOOD                    |

### Llama3.2 2B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 12GB | TOOD                     | TOOD                   | TOOD                    |
| 16GB | TOOD                     | TOOD                   | TOOD                    |
| 24GB | TOOD                     | TOOD                   | TOOD                    |
| 48GB | TOOD                     | TOOD                   | TOOD                    |
| 80GB+| TOOD                     | TOOD                   | TOOD                    |

### Qwen3 4B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 16GB | TOOD                     | TOOD                   | TOOD                    |
| 24GB | TOOD                     | TOOD                   | TOOD                    |
| 48GB | TOOD                     | TOOD                   | TOOD                    |
| 80GB+| TOOD                     | TOOD                   | TOOD                    |

### Qwen3 8B
| VRAM  | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ----- | :----------------------: | :--------------------: | :---------------------: |
| 24GB  | TOOD                     | TOOD                   | TOOD                    |
| 32GB  | TOOD                     | TOOD                   | TOOD                    |
| 48GB  | TOOD                     | TOOD                   | TOOD                    |
| 80GB+ | TOOD                     | TOOD                   | TOOD                    |


### 2. Additional Tips
* **Activation Checkpointing** significantly reduces VRAM usage but incurs roughly 20-30% longer training times.
* **Gradient Checkpointing** slightly reduces memory with minimal impact on speed.
* **Train Length Truncation** controls maximum input token length, with shorter lengths substantially reducing VRAM requirements.
* Adjust batch sizes carefully—larger batch sizes may require lower truncation lengths.


---