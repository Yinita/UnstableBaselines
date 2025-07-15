# Unstable Baselines Documentation

> **Version:** 0.2 · **Last Updated:** 2025-07-15
>
> This release introduces a dedicated **sampler** layer (`samplers/`), a composable **runtime** builder (`unstable.build()`), and separate **REINFORCE** / **A2C** learners.  `ModelPool` has been renamed **ModelRegistry**, and the data layer now exposes both **StepBuffer** *and* **EpisodeBuffer**.

---
```
Lines of Code per Release
-------------------------
0.1.0  | ######################     1,144       -> initial release
0.2.0  | ########################   1,288       -> added A2C, runtime object, environment scheduling
```
---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   * Installation
   * Minimal Script
   * Standard Script
3. [Architecture Overview](#architecture-overview)
4. [Core Modules](#core-modules)
5. [Reward Transformations](#reward-transformations)
6. [Algorithms](#algorithms)
7. [Configuration Reference](#configuration-reference)

---

## Introduction <a id="introduction"></a>

The key piece holding **Unstable Baselines** together is the **Collector**.  It maintains a pool of `num_train_workers` and `num_eval_workers` games in flight.  A **GameScheduler** decides *what* to run next by querying the **EnvSampler** (which environment?) and **ModelSampler** (which opponent?).  When a game finishes, trajectories stream into a replay **Buffer**, metrics go to the **Tracker**, and TrueSkill updates flow back to the **ModelRegistry**. Once enough samples are available in the **Buffer**, the **Learner** (currently either **REINFORCELeanrner** or **A2CLearner**) pulls them and trains. The new checkpoint will be added to the **ModelRegistry**.

---

## Getting Started <a id="getting-started"></a>

### Installation

```bash
pip install unstable-rl
```

### Minimal Script
TO offer and easier starting off point, in `v0.2.0` we added a runner that handles all necessary initializations etc. Here is an example script for training `Qwen/Qwen3-1.7B-Base` on `SimpleTak-v0-train` via mirror self-play and evaluating it on `SimpleTak-v0-train` and `KuhnPoker-v0-train` against **google/gemini-2.0-flash-lite-001** (our default fixed opponent).

```python
import unstable

run = unstable.build(
    model_name = "Qwen/Qwen3-1.7B-Base",
    train_envs = [unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs")],
    eval_envs = [
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs")
    ]
)
run.start(learning_steps=200, num_collection_workers=256, num_eval_workers=16)
```
The `unstable.build(...)` setup includes a lot of default choices that are somewhat hidden. To make using this easier, here is a table of parameters, expected formats and the default choices:

| **Parameter**                | **Type / Accepted values**         | **Default**                                                 | **What it does / notes**                                                                                                    |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------|
| `model_name`                 | `str` (HF repo, GGUF path, etc.)   | **required**                                                | Base LM to fine-tune / RL-train.                                                                                            |
| `train_envs`                 | `Sequence[TrainEnvSpec]`           | **required**                                                | Environments used for data collection / learning.                                                                           |
| `eval_envs`                  | `Sequence[EvalEnvSpec] \| None`    | `None`                                                      | Optional evaluation envs; if omitted, no eval runs.                                                                         |
| `env_sampling_strategy`      | `"random"` *(only option for now)* | `"random"`                                                  | Chooses the env sampler. Maps to `UniformRandomEnvSampler`.                                                                 |
| `opponent_sampling_strategy` | `"none"`, `"mirror"`, `"fixed"`    | `"none"`                                                    | How collectors pick opponent models.<br/>• **none**/**mirror** → self-play<br/>• **fixed** → sample from `fixed_opponents`. |
| `fixed_opponents`            | `Sequence[str]`                    | `["google/gemini-2.0-flash-lite-001"]`                      | Only used when `opponent_sampling_strategy="fixed"`.                                                                        |
| `algorithm`                  | `"reinforce"` or `"a2c"`           | `"reinforce"`                                               | Chooses learner class and buffer shape.                                                                                     |
| `max_train_len`              | `int \| None`                      | `None`                                                      | Truncation length for *training* prompts; fed into `vllm_config["max_tokens"]`.                                             |
| `max_generation_len`         | `int`                              | `4096`                                                      | Truncation length for *inference* continuation and Dr. GRPO Trick.                                                          |
| `batch_size`                 | `int`                              | `384`                                                       | Global batch per learner update (pulled from buffer).                                                                       |
| `mini_batch_size`            | `int`                              | `1`                                                         | Micro-batch size inside each update step.                                                                                   |
| `learning_rate`              | `float`                            | `1e-5`                                                      | AdamW base LR.                                                                                                              |
| `gradient_clipping`          | `float`                            | `0.2`                                                       | Global‐norm clip.                                                                                                           |
| `activation_checkpointing`   | `bool`                             | `True`                                                      | Enables selective `torch.utils.checkpoint` on forwards.                                                                     |
| `gradient_checkpointing`     | `bool`                             | `True`                                                      | Turns on HF-style gradient ckpt in transformer blocks.                                                                      |
| `use_trainer_cache`          | `bool`                             | `False`                                                     | Re-use a cached HF `Trainer` between restarts.                                                                              |
| `buffer_size`                | `int \| None`                      | `batch_size × 2`                                            | Capacity of the shared replay buffer.                                                                                       |
| `lora_config`                | `dict \| None`                     | `None` → falls back to **\_DEFAULT\_LORA\_CFG** (see below) | LoRA hyper-params applied to the model.                                                                                     |
| `vllm_config`                | `dict \| None`                     | `None` → auto-built by **\_default\_vllm\_cfg**             | Passed straight to `Collector` for vLLM engine.                                                                             |
| `wandb_project`              | `str`                              | `"UnstableBaselines"`                                       | Which Weights & Biases project to log into.                                                                                 |

```python
_DEFAULT_LORA_CFG = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj", 
                       "gate_proj","up_proj","down_proj"],
}
```
```python
def _default_vllm_cfg(model_name, lora_cfg, max_generation_len):
    return {
        "model_name": model_name,
        "temperature": 0.6,
        "max_tokens": max_generation_len,   # <- None means vLLM will infer
        "max_parallel_seq": 128,
        "max_loras": 8,
        "lora_config": lora_cfg,
        "max_model_len": 8192,
    }

```


### Standard Script

If you are looking for a bit more control and flexibility, here is how we usually run the code:

```python
import time, ray, unstable
import unstable.reward_transformations as retra

COLLECTION_WORKERS = 384
EVALUATION_WORKERS = 16
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-4B-Base"
BATCH_SIZE = 384
MINI_BATCH_SIZE = 1
BUFFER_SIZE = 384*2
LR = 1e-5
GRAD_CLIP = 0.2
MAX_TRAIN_SEQ_LEN = None
MAX_GENERATION_LENGTH = 4096 

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}

# Ray init
ray.init(namespace="unstable")  

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"), # if num_players == num_actors, it's mirror self-play and no opponents will be sampled
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs"),
])

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

# initialize model registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

# initialize model sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

# Data Buffer
step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# initialize the collector
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
)

# initialize the learner
learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(max_train_len=MAX_TRAIN_SEQ_LEN, max_generation_len=MAX_GENERATION_LENGTH))


try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
```

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