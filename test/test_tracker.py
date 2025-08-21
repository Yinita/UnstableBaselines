import ray
from unstable.trackers import Tracker

def main():
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True, log_to_driver=True)
    try:
        t = Tracker.remote(run_name="tracker-smoke-test", wandb_project=None)
        # Log learner info with mixed types
        ray.get(t.log_learner.remote({
            "loss": 0.123,
            "lr": 1e-4,
            "scalars": {"grad_norm": 0.9, "gpu_mem": 12.3},
            "nonscalar": [1,2,3],
        }))
        # Log inference stats
        ray.get(t.log_inference.remote("actor0", [0], {"tok_s": 1000.0, "latency_ms": 12.0}))
        # Simulate collection metrics
        # Using a minimal PlayerTrajectory-like object
        class DummyTraj:
            def __init__(self):
                self.final_reward = 1
                self.pid = 0
                self.obs = ["obs"]
                self.actions = ["act"]
                self.format_feedbacks = [{"correct_answer_format": 1, "invalid_move": 0}]
                self.num_turns = 1
        ray.get(t.add_player_trajectory.remote(DummyTraj(), "DummyEnv"))
        # Force internal flush
        info = ray.get(t.get_interface_info.remote())
        assert "gpu_tok_s" in info
        print("Tracker smoke test OK.")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
