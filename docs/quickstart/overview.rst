# Overview

`UnstableBaselines` marries three external pillars:

* **Ray** provides elastic, fault-tolerant distributed execution.
* **vLLM** enables high-throughput language-model inference with *LoRA hot-swap*.
* **TextArena** supplies a library of text-based competitive games (1 v 1, team, and free-for-all).

The default training recipe is **mirror self-play** with on-policy REINFORCE, but every moving part—sampling strategy, reward shaping, update rule—can be swapped independently.

Why “*Unstable*”? Because research is messy.  The codebase stays under **1.5 kLoC**, avoids hidden globals, and treats *hackability* as its primary feature.

Core loops (Collector → StepBuffer → Learner).
