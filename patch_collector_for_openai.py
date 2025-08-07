"""
Patch for UnstableBaselines collector to support OpenAI agents as opponents.
This module monkey-patches the run_game function to handle custom OpenAI agents.
"""

import ray
import sys
import os
from typing import Dict, Any

# Add the UnstableBaselines directory to Python path
sys.path.append('/home/aiscuser/mindgames/UnstableBaselines')
from defined_agents import OpenAIAgent

# Import necessary UnstableBaselines modules
import unstable.collector as collector_module
from unstable._types import GameSpec, GameInformation, PlayerTrajectory
from unstable.actor import VLLMActor
from unstable.collector import CallableActorWrapper, OBSERVATION_FORMATTING, ACTION_EXTRACTION
import textarena as ta


def create_openai_agent_for_opponent(opponent_name: str, openai_config: Dict[str, Any]):
    """Create an OpenAI agent based on the opponent name and configuration."""
    if opponent_name.startswith("openai-"):
        # Extract model name from opponent name (e.g., "openai-gpt4o-mini" -> "gpt-4o-mini")
        model_name = openai_config.get("model_name", "gpt-4o-mini")
        return OpenAIAgent(
            model_name=model_name,
            api_key=openai_config.get("api_key"),
            base_url=openai_config.get("base_url"),
            verbose=openai_config.get("verbose", False)
        )
    else:
        # Fall back to OpenRouter for other opponents
        return ta.agents.OpenRouterAgent(opponent_name)


def patched_run_game_impl(game_spec: GameSpec, actor: VLLMActor, openai_config: Dict[str, Any] = None):
    """
    Implementation of patched run_game that supports OpenAI agents as opponents.
    """
    if openai_config is None:
        openai_config = {}
    
    game_information = GameInformation(
        game_idx=game_spec.game_idx, 
        eval_model_pid=game_spec.eval_model_pid, 
        eval_opponent_name=game_spec.eval_opponent_name
    )
    
    # Build agents with support for OpenAI agents
    agents = {}
    for agent_spec in game_spec.agent_specs:
        agent_info = {
            "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None,
            "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
        }
        
        # Determine the model/agent to use
        if agent_spec.openrouter_name is None:
            # This is a checkpoint/training model
            agent_info["model"] = CallableActorWrapper(
                actor=actor, 
                lora_path=agent_spec.lora_path, 
                obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template], 
                extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn]
            )
        else:
            # This is an opponent - check if it's an OpenAI agent or OpenRouter
            agent_info["model"] = create_openai_agent_for_opponent(
                agent_spec.openrouter_name, 
                openai_config
            )
        
        agents[agent_spec.pid] = agent_info
    
    # Initialize environment and run the game
    env = ta.make(game_spec.env_id)
    env.reset(num_players=len(agents), seed=game_spec.seed)
    env.state.error_allowance = 0
    turn = 0
    
    while True:
        pid, obs = env.get_observation()
        
        # Get model (or opponent) action
        if agents[pid]["traj"] is None:  
            # Fixed opponent
            raw = extracted = agents[pid]["model"](obs)
        else:  
            # Training model
            raw, extracted, prompt, format_feedback = agents[pid]["model"].act_full(obs)
        
        # Execute the action & increment turn counter
        done, step_info = env.step(extracted)
        turn += 1
        
        # General tracking
        game_information.pid.append(pid)
        game_information.obs.append(obs)
        game_information.full_actions.append(raw)
        game_information.extracted_actions.append(extracted)
        game_information.step_infos.append(step_info)
        game_information.names[pid] = agents[pid]["name"]
        
        # Player specific tracking
        if agents[pid]["traj"] is not None:
            agents[pid]["traj"].obs.append(obs)
            agents[pid]["traj"].actions.append(raw)
            agents[pid]["traj"].extracted_actions.append(extracted)
            format_feedback["invalid_move"] = False
            agents[pid]["traj"].format_feedbacks.append(format_feedback)
            agents[pid]["traj"].step_infos.append(step_info)
        
        if done:
            break
    
    # Finalize game
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"] is not None:
            agents[pid]["traj"].final_reward = final_rewards[pid]
            agents[pid]["traj"].game_info = game_info[pid]
            agents[pid]["traj"].num_turns = turn
            if game_info[pid]["invalid_move"] and agents[pid]["traj"] is not None:
                agents[pid]["traj"].format_feedbacks[-1]["invalid_move"] = True
    
    game_information.final_rewards = final_rewards
    game_information.num_turns = turn
    game_information.game_info = game_info
    
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"] is not None]


def patch_collector_for_openai(openai_config: Dict[str, Any]):
    """
    Monkey patch the collector module to support OpenAI agents.
    
    Args:
        openai_config: Configuration for OpenAI agent including model_name, api_key, base_url, etc.
    """
    print("Patching UnstableBaselines collector for OpenAI agent support...")
    
    # Create a new remote function that wraps our implementation with the config
    @ray.remote(num_cpus=0)
    def run_game_with_openai_support(game_spec: GameSpec, actor: VLLMActor):
        return patched_run_game_impl(game_spec, actor, openai_config)
    
    # Replace the original run_game function
    collector_module.run_game = run_game_with_openai_support
    
    print("Collector patched successfully!")
    return True


if __name__ == "__main__":
    # Example usage
    openai_config = {
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "verbose": True
    }
    
    patch_collector_for_openai(openai_config)
    print("Patch applied successfully!")
