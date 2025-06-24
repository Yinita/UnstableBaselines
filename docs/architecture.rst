# System Architecture

.. \_arch-diagram:

## ASCII block diagram

\::

                                                ┌───────────────┐
                                                │               │
                                                │   Algorithm   │
                                                │               │
                                                └───────────────┘
                                                        ▲        
                                                        │ Get Loss &
                                                        │ update weights
                                                        ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │    Register new lora      │               │
    │   Model Pool  │◀──────────────────────────│    Learner    │
    │               │       checkpoint          │               │
    └───────────────┘                           └───────────────┘
           ▲ │                                         ▲ │ 
           │ │ Sample                        If enough │ │ Check if enough
    Update │ │ Opponent                     data, pull │ │ data for training
 Trueskill │ │                          the next batch │ │ is available
           │ ▼                                         │ ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │     Process and store     │               │
    │   Collector   │──────────────────────────▶│   StepBuffer  │
    │               │  collected Trajectories   │               │
    └───────────────┘                           └───────────────┘
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

## Data-flow summary

\#. **Collector** rolls games with the latest learner checkpoint vs. an opponent sampled by **ModelPool** (mirror, lagged, TrueSkill-based, etc.).
\#. Completed trajectories pass to **StepBuffer**, which applies reward transforms and holds a sliding window of training steps.
\#. **Learner** periodically drains a batch, performs back-prop via the chosen **Algorithm** (default: REINFORCE), saves a new LoRA checkpoint, and registers it with **ModelPool**.
\#. **Tracker** logs metrics to Weights & Biases and exposes state for the **Terminal Interface** dashboard.
