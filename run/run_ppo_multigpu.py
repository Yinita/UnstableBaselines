import ray, time, torch, os

# Set PyTorch CUDA allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from unstable.agents.ppo import PPO
from unstable.envs.dummy import DummyEnv

# Configuration for the multi-GPU PPO run
cfg = {
    'environment': {
        'name': 'dummy',
        'max_steps': 100,
        'num_envs': 4,
    },
    'agent': {
        'name': 'ppo',
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_clip': 0.2,
        'c1': 1.0,  # Value loss coefficient
        'c2': 0.01, # Entropy coefficient
        'lr': 1e-4,
        'lr_critic': 1e-4,
        'num_gpus': 2, # Request 2 GPUs for the learner
        'num_gpus_per_worker': 0.1, # GPUs for collectors
        'num_sgd_iter': 10,
        'train_batch_size': 4096,
        'rollout_fragment_length': 2048,
        'sgd_minibatch_size': 512,
        'max_seq_len': 1024, # Reduce memory usage
        'model': {
            'model_name': 'gpt2',
            'lora_rank': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
        }
    }
}

def main():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=4, ignore_reinit_error=True)
    print("Ray initialized.")
    print(f"Available resources: {ray.available_resources()}")

    # Create the PPO agent
    agent = PPO(cfg=cfg, env=DummyEnv)

    # Start training
    print("Starting PPO training with multi-GPU setup...")
    start_time = time.time()
    agent.train()
    end_time = time.time()

    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Clean up
    agent.shutdown()
    ray.shutdown()
    print("Ray shut down.")

if __name__ == '__main__':
    main()
