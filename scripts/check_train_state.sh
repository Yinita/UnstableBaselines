python analysis/compute_cumwin_eval.py       --base /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-22/00-24-49/MixedPlay-Qwen3-8B-Codenames-v0-train-1755822283/logs/samples   --save_csv analysis/cumwin_eval_summary.csv   --plot_path analysis/cumwin_eval.png

python analysis/compute_cumwin_training.py   --base /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-22/00-24-49/MixedPlay-Qwen3-8B-Codenames-v0-train-1755822283/training_data   --save_csv analysis/cumwin_training_summary.csv   --plot_path analysis/cumwin_training.png


python analysis/compute_cumwin_training.py   --base /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-28/04-21-01/ppo-4o-4b-0828-v1/training_data   --save_csv analysis/ppo-4o-4b-0828-train_summary.csv   --plot_path analysis/ppo-4o-4b-0828-train.png

python analysis/compute_cumwin_eval.py   --base /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-28/04-21-01/ppo-4o-4b-0828-v1/logs/samples   --save_csv analysis/ppo-4o-4b-0828-eval_summary.csv   --plot_path analysis/ppo-4o-4b-0828-eval.png
