import unstable
import unstable.reward_transformations as retra

# initialize the tracker to keep wandb up to date and print as necessary
tracker = unstable.Tracker()


# compose the reward transformations
final_reward_transformation = retra.ComposeFinalRewardTransforms([
    retra.RoleAdvantageByEnvFormatter(), # normalize rewards for role advantage # TODO worth moving to step?
])
step_reward_transformation = retra.ComposeStepRewardTransforms([
    retra.RewardForThinkTags(reward=args.format_reward_think), # +0.25 for correct <think></think> tags
    retra.PenaltyForInvalidMove(reward=args.format_reward_valid_move, penalty=args.format_penalty_invalid_move), 
])
sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([
    retra.NormalizeRewardsByEnv(z_score=True) # normalize the sampled batch
])

# initialize the StepBuffer (used to hold and sample from collected traces)
step_buffer = unstable.StepBuffer(
    max_buffer_size = max_buffer_size,
    final_reward_transformation = final_reward_transformation,
    step_reward_transformation = step_reward_transformation,
    sampling_reward_transformation = sampling_reward_transformation,
)

# TODO buffer needs to build the output dir and communicate it to everybody else
# TODO buffer needs to hold the logging object (such that both the collector and learner can log stuff via it)


# initialize Collector
collector = unstable.Collector(
    num_actors = 2, step_buffer = step_buffer,
    training_envs = [("SimpleTak-v0-train", 2, "zero_sum_game_play")],
    evaluation_envs = [("SimpleTak-v0-train", 2, "zero_sum_game_play"), ("AIME24-v0", 1, "standard_reasoning")],
    opponent_strategy = "self-play", vllm_dict = {},
    num_collection_workers = 384, num_evaluation_workers = 32
)


# initialize the model and the learner
learning_algorithm = unstable.algorithms.Reinforce(
    model=model, tokenizer=tokenizer, device=device
)


# # initialize the Trainer
# learner = unstable.Learner(
#     num_learners = 1, 
#     step_buffer = step_buffer,
#     model_name = "Qwen/Qwen3-4B-base"
#     algorithm = learning_algorithm,
#     iterations = 1000
# )


# start both loops (one for Collector, one for Trainer)
collector.collect() # will keep collecting until trainer is done with the designated number of iterations
# trainer.train() # will keep training when data is available until "iterations" many update steps



# TODO split up the rewards when saving the training data 