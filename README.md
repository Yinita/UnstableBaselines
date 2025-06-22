# unstable-baselines

[![Status](https://img.shields.io/badge/status-WIP-orange?style=for-the-badge&label=Project%20Status)](#)
[![TextArena](https://img.shields.io/badge/TextArena-v0.6.9-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/LeonGuertler/TextArena)
[![Discord](https://img.shields.io/discord/1257951838322561075?color=7289DA&label=TextArena%20Discord&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/KPacHzK23e)


> **unstable‑baselines** is an **experimental, asynchronous, online reinforcement‑learning framework**
> for rapid prototyping of *multi‑turn / multi‑agent* algorithms on
> [TextArena](https://github.com/LeonGuertler/TextArena) environments.
>
> The main focus on unstable baselines is to enable fast prototyping/research. For something a bit more production ready we recomment to use [oat](https://github.com/sail-sg/oat) or [verifiers](https://github.com/willccbb/verifiers)
>
> **Work in progress — interfaces will change.**

---

## Key Features
* **Asynchronous collection & learning** – actors generate data while learners train.
* **Multi‑agent, multi‑turn** focus with self‑play or fixed opponents.
* **LoRA‑first** fine‑tuning workflow for fast, lightweight updates.
* **Composable reward transforms** at step, final, and sampling stages.


## Structure
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


## Installation

```bash
# build TextArena v0.6.9 (until it’s on PyPI)
git clone https://github.com/LeonGuertler/TextArena.git
cd TextArena
git checkout v0.6.9
pip install -e .
cd ..

# clone the repo
git clone https://github.com/LeonGuertler/UnstableBaselines.git
cd UnstableBaselines

# install Python dependencies
pip install -r requirements.txt
```

## Quick Start
Starting the main script:
```bash
python3 main.py
```

If you want to monitor key metrics (in addition to logging them via W&B) you can run the following in a seperate terminal:
```bash
python3 -m unstable.terminal_interface
```

## Motivation for LoRA vs full fine-tuning
TODO


## Collaboration

Developed in partnership with [PlasticLabs](https://plasticlabs.ai/).
