# unstable-baselines

[![Status: WIP](https://img.shields.io/badge/status-WIP-orange?style=for-the-badge)](#)
[![Async Online RL](https://img.shields.io/badge/async-online%20RL-blue?style=for-the-badge)](#)
[![TextArena v0.6.9](https://img.shields.io/badge/TextArena-0.6.9-blue?style=for-the-badge\&logo=github)](https://github.com/LeonGuertler/TextArena)
[![PlasticLabs](https://img.shields.io/badge/PlasticLabs-ai-purple?style=for-the-badge)](https://plasticlabs.ai/)
[![Discord](https://img.shields.io/discord/1257951838322561075?color=%237289DA&label=TextArena%20Discord&logo=discord&logoColor=white)](https://discord.gg/KPacHzK23e)
[![MIT License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

> **unstable‑baselines** is an **experimental, asynchronous, online reinforcement‑learning framework**
> for rapid prototyping of *multi‑turn / multi‑agent* algorithms on
> [TextArena](https://github.com/LeonGuertler/TextArena) environments.
>
> **Work in progress — interfaces will change.**

---

## Key Features

* **Asynchronous collection & learning** – actors generate data while learners train.
* **Multi‑agent, multi‑turn** focus with self‑play or fixed opponents.
* **LoRA‑first** fine‑tuning workflow for fast, lightweight updates.
* **Composable reward transforms** at step, final, and sampling stages.
* **One‑command launch** (`bash run.sh`) with built‑in evaluation and W\&B logging.

## Collaboration

Developed in partnership with [PlasticLabs](https://plasticlabs.ai/).

## Installation

```bash
# clone the repo
git clone https://github.com/LeonGuertler/unstable-baselines.git
cd unstable-baselines

# install Python dependencies
pip install -r requirements.txt

# build TextArena v0.6.9 (until it’s on PyPI)
git clone https://github.com/LeonGuertler/TextArena.git
cd TextArena
git checkout v0.6.9
pip install -e .
cd ..
```

## Quick Start

```bash
# fine‑tune Qwen3 on SimpleTak with 7 actors and 1 learner
bash run.sh
```

Adjust `--num_actors`, `--num_learners`, and the environment lists inside `run.sh` to match your GPU setup.

## Project Layout

```text
unstable-baselines/
|-- algorithms/
|   |-- reinforce.py
|   |-- reinforce_kl.py
|   |-- ppo.py
|   `-- ...
|-- actors/
|   `-- vllm_actor.py
|-- learners/
|   |-- single_node_distributed.py
|   `-- lora_utils.py
|-- reward_transformations/
|-- utils/
|-- trajectory_buffer.py
|-- unstable.py   # main entrypoint
`-- run.sh        # example launch script
```

## Roadmap

* [ ] PPO with GAE
* [ ] GRPO (branching and efficient)
* [ ] Multi‑Node training
* [ ] Single‑GPU training
* [ ] Rich dashboards
