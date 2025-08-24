"""
Patch for UnstableBaselines collector to support OpenAI agents as opponents.
This module monkey-patches the run_game function to handle custom OpenAI agents.
"""

import ray
import logging
import sys
import os
import contextlib
from typing import Dict, Any

# Add the UnstableBaselines directory to Python path
sys.path.append('/home/aiscuser/mindgames/UnstableBaselines')
from defined_agents import OpenAIAgent, OpenRouterAgent

# Import necessary UnstableBaselines modules
import unstable.collector as collector_module
from unstable._types import GameSpec, GameInformation, PlayerTrajectory
from unstable.actor import VLLMActor
from unstable.collector import CallableActorWrapper, OBSERVATION_FORMATTING, ACTION_EXTRACTION
import textarena as ta

# --- Monkey patch: Redirect TextArena's OpenRouterAgent to local OpenAIAgent ---
# Some TextArena envs instantiate a GM via ta.agents.OpenRouterAgent(model_name="openai/gpt-4o"),
# which requires OPENROUTER_API_KEY. For single-player/local runs we want to avoid external calls
# and use our local OpenAIAgent("gpt-4o") instead. We provide a lightweight shim that matches
# the agent API used by environments (notably an act or act_full interface).

class _LocalGM_OpenAIShim:
    """Shim that mimics OpenRouterAgent but internally uses our OpenAIAgent.

    It forwards calls to OpenAIAgent("gpt-4o") with the same act/act_full signature
    expected by TextArena envs. This avoids needing OPENROUTER_API_KEY.
    """

    def __init__(self, model_name: str = "openai/gpt-4o", *args, **kwargs):
        # Normalize: TextArena passes "openai/gpt-4o"; our agent expects "gpt-4o"
        local_name = model_name.split("/")[-1] if model_name else "gpt-4o"
        # Quiet console by default to reduce noise
        self._agent = OpenAIAgent(model_name=local_name, verbose=False, api_key=os.getenv("OPENAI_API_KEY", ""), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    # Some envs may call act(obs) returning a string; others call act_full(obs) returning 5-tuple
    def act(self, observation: str):
        # Fallback to act_full and return the raw text
        raw, extracted, prompt, format_feedback, logp = self.act_full(observation)
        return raw

    # Make the shim callable like a function (TextArena does gamemaster(prompt))
    def __call__(self, observation: str) -> str:
        return self.act(observation)

    def act_full(self, observation: str):
        # Our OpenAIAgent exposes act_full already; if not, emulate minimal behavior
        if hasattr(self._agent, "act_full"):
            return self._agent.act_full(observation)
        # Minimal fallback: produce raw=completion, extracted=None, prompt=None, format_feedback=True, logp=[]
        raw = self._agent.act(observation) if hasattr(self._agent, "act") else ""
        return raw, None, None, True, []

# Apply the monkey patch early so that subsequent ta.make(...) picks it up
try:
    # Only patch if the attribute exists to avoid breaking other versions
    if hasattr(ta, "agents") and hasattr(ta.agents, "OpenRouterAgent"):
        ta.agents.OpenRouterAgent = _LocalGM_OpenAIShim  # type: ignore
except Exception as _patch_exc:
    logging.warning(f"Failed to monkey-patch OpenRouterAgent: {_patch_exc}")


def create_openai_agent_for_opponent(opponent_name: str, openai_config: Dict[str, Any]):
    """Create an OpenAI agent based on the opponent name and configuration."""
    # logging.info(f"Creating agent for opponent: {opponent_name}")
    if openai_config.get("verbose", False):
        logging.info(f"Creating agent for opponent: {opponent_name}")
    if opponent_name.startswith("openai-"):
        # 提取模型名称
        model_name = opponent_name[len("openai-"):]
        logging.info(f"Extracted model name: {model_name}")
        
        try:
            # 创建OpenAI代理
            agent = OpenAIAgent(
                model_name=model_name,
                verbose=openai_config.get("verbose", False)
            )
            # logging.info(f"Successfully created OpenAI agent with model: {model_name}")
            # 确认agent有act_full方法
            if not hasattr(agent, 'act_full'):
                logging.error(f"OpenAI agent does not have act_full method, adding compatibility wrapper")
                original_call = agent.__call__
                def act_full_wrapper(observation):
                    raw = original_call(observation)
                    return raw, raw, observation, {}, 0.0
                agent.act_full = act_full_wrapper
            return agent
        except Exception as e:
            logging.error(f"Failed to create OpenAI agent: {e}", exc_info=True)
            raise
    else:
        # 回退到OpenRouter
        # logging.info(f"Using OpenRouter agent for: {opponent_name}")
        agent = OpenRouterAgent(opponent_name)
        # 确认agent有act_full方法
        if not hasattr(agent, 'act_full'):
            logging.error(f"OpenRouter agent does not have act_full method, adding compatibility wrapper")
            original_call = agent.__call__
            def act_full_wrapper(observation):
                raw = original_call(observation)
                return raw, raw, observation, {}, 0.0
            agent.act_full = act_full_wrapper
        return agent

def patched_run_game_impl(game_spec: GameSpec, actor: VLLMActor, openai_config: Dict[str, Any] = None):
    """
    Implementation of patched run_game that supports OpenAI agents as opponents.
    """
    # 验证game_spec是否有效
    if game_spec is None:
        error_msg = "ERROR: game_spec is None in patched_run_game_impl"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    # 记录详细的game_spec信息以便调试
    # logging.info(f"Running game with spec: game_idx={game_spec.game_idx}, "
    #             f"eval_model_pid={game_spec.eval_model_pid}, "
    #             f"eval_opponent_name={game_spec.eval_opponent_name}, "
    #             f"env_id={game_spec.env_id}, seed={game_spec.seed}")
    
    if openai_config is None:
        openai_config = {}
        # logging.warning("No OpenAI config provided, using empty dict")
    
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
        # 统一使用act_full方法，确保返回5元组
        try:
            # logging.info(f"[TRAJ_DEBUG] 调用agent.act_full pid={pid}, agent_type={type(agents[pid]['model']).__name__}, obs_len={len(obs)}")
            raw, extracted, prompt, format_feedback, logp = agents[pid]["model"].act_full(obs)
            # logging.info(f"[TRAJ_DEBUG] act_full成功返回 pid={pid}, raw_len={len(raw)}, extracted_len={len(extracted)}")
        except Exception as e:
            # logging.error(f"[TRAJ_DEBUG] act_full调用失败 pid={pid}, error={e}", exc_info=True)
            raise
        
        # Execute the action & increment turn counter
        try:
            # logging.info(f"[TRAJ_DEBUG] 执行动作 env.step pid={pid}, action_len={len(extracted)}")
            done, step_info = env.step(extracted)
            # logging.info(f"[TRAJ_DEBUG] 动作执行完成 pid={pid}, done={done}")
            turn += 1
        except Exception as e:
            # logging.error(f"[TRAJ_DEBUG] 动作执行失败 pid={pid}, error={e}", exc_info=True)
            raise
        
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
            agents[pid]["traj"].logps.append(logp)
            if format_feedback is None:
                format_feedback = {}
            format_feedback["invalid_move"] = False
            agents[pid]["traj"].format_feedbacks.append(format_feedback)
            agents[pid]["traj"].step_infos.append(step_info)
            if turn % 100 == 0:
                ol = len(agents[pid]["traj"].obs); al = len(agents[pid]["traj"].actions); ll = len(agents[pid]["traj"].logps)
                logging.getLogger("collector").info(f"turn={turn} pid={pid} lengths obs={ol} acts={al} logps={ll}")
        
        if done:
            break
    
    # Finalize game
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"] is not None:
            agents[pid]["traj"].final_reward = final_rewards[pid]
            agents[pid]["traj"].game_info = game_info[pid]
            agents[pid]["traj"].num_turns = turn
            if game_info[pid].get("invalid_move", False) and agents[pid]["traj"] is not None:
                if agents[pid]["traj"].format_feedbacks:
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
    # print("Patching UnstableBaselines collector for OpenAI agent support...")
    
    # Create a new remote function that wraps our implementation with the config
    @ray.remote(num_cpus=0)
    def run_game_with_openai_support(game_spec: GameSpec, actor: VLLMActor):
        # 验证参数
        if game_spec is None:
            error_msg = "ERROR: game_spec is None in run_game_with_openai_support"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        if actor is None:
            error_msg = "ERROR: actor is None in run_game_with_openai_support"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        # Optional quiet mode to suppress console noise from the environment and agents
        quiet = openai_config.get("quiet_console", False)
        if quiet:
            # Reduce logging verbosity in this worker
            logging.getLogger().setLevel(logging.WARNING)

        logging.info(f"Starting game with openai_support, game_spec type: {type(game_spec)}, actor type: {type(actor)}")
        try:
            if quiet:
                # Silence stdout/stderr (prints from game/env) during gameplay
                with open(os.devnull, 'w') as devnull, \
                     contextlib.redirect_stdout(devnull), \
                     contextlib.redirect_stderr(devnull):
                    return patched_run_game_impl(game_spec, actor, openai_config)
            else:
                return patched_run_game_impl(game_spec, actor, openai_config)
        except Exception as e:
            logging.error(f"Error in run_game_with_openai_support: {e}", exc_info=True)
            raise
    
    # Replace the original run_game function
    collector_module.run_game = run_game_with_openai_support
    
    # print("Collector patched successfully!")
    return True


if __name__ == "__main__":
    # Example usage
    openai_config = {
        "model_name": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "verbose": True
    }
    
    patch_collector_for_openai(openai_config)
    print("Patch applied successfully!")