# Troubleshooting

## Runtime hiccups

* **Learner GPU OOM** → try `activation_checkpointing=False` or reduce `MAX_TRAIN_SEQ_LEN`.
* **vLLM deadlock** detected by *Actor* logs → set `EngineArgs(enforce_eager=True)` for debugging.