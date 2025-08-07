"""
Custom agent integration for UnstableBaselines framework.
This module provides a way to integrate custom agents (like OpenAIAgent) 
as opponents in the training process.
"""

import ray
import sys
import os
from typing import Dict, Any, Optional
from unstable.utils import setup_logger

# Add the UnstableBaselines directory to Python path to import defined_agents
sys.path.append('/home/aiscuser/mindgames/UnstableBaselines')
from defined_agents import OpenAIAgent, OpenRouterAgent, HumanAgent


@ray.remote
class CustomAgentRegistry:
    """Registry for managing custom agents that can be used as opponents."""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.logger = setup_logger("custom_agent_registry", ray.get(self.tracker.get_log_dir.remote()))
        self._agents: Dict[str, Any] = {}
    
    def register_openai_agent(self, name: str, model_name: str, api_key: Optional[str] = None, 
                             base_url: Optional[str] = None, **kwargs):
        """Register an OpenAI agent as a custom opponent."""
        try:
            agent = OpenAIAgent(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
            self._agents[name] = {
                'type': 'openai',
                'agent': agent,
                'model_name': model_name
            }
            self.logger.info(f"Registered OpenAI agent: {name} with model {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register OpenAI agent {name}: {e}")
            return False
    
    def register_openrouter_agent(self, name: str, model_name: str, **kwargs):
        """Register an OpenRouter agent as a custom opponent."""
        try:
            agent = OpenRouterAgent(model_name=model_name, **kwargs)
            self._agents[name] = {
                'type': 'openrouter',
                'agent': agent,
                'model_name': model_name
            }
            self.logger.info(f"Registered OpenRouter agent: {name} with model {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register OpenRouter agent {name}: {e}")
            return False
    
    def get_agent(self, name: str):
        """Get a registered agent by name."""
        if name in self._agents:
            return self._agents[name]['agent']
        else:
            self.logger.warning(f"Agent {name} not found in registry")
            return None
    
    def list_agents(self):
        """List all registered agents."""
        return {name: info['model_name'] for name, info in self._agents.items()}
    
    def call_agent(self, name: str, observation: str) -> str:
        """Call a registered agent with an observation."""
        agent = self.get_agent(name)
        if agent:
            try:
                response = agent(observation)
                return response
            except Exception as e:
                self.logger.error(f"Error calling agent {name}: {e}")
                return "ERROR: Agent call failed"
        else:
            return "ERROR: Agent not found"


class CustomAgentIntegration:
    """Integration helper for using custom agents in UnstableBaselines."""
    
    def __init__(self, custom_agent_registry):
        self.custom_agent_registry = custom_agent_registry
    
    @staticmethod
    def create_custom_agent_spec(pid: int, agent_name: str, prompt_template: str, 
                                action_extraction_fn: str = "default"):
        """Create an AgentSpec for a custom agent."""
        from unstable._types import AgentSpec
        return AgentSpec(
            pid=pid,
            kind="custom",  # New kind for custom agents
            collect_data=False,
            lora_path=None,
            openrouter_name=agent_name,  # Reuse this field for agent name
            prompt_template=prompt_template,
            action_extraction_fn=action_extraction_fn
        )
    
    def handle_custom_agent_action(self, agent_name: str, observation: str) -> str:
        """Handle action generation for custom agents."""
        return ray.get(self.custom_agent_registry.call_agent.remote(agent_name, observation))


# Monkey patch to extend the game runner to support custom agents
def patch_run_game_for_custom_agents():
    """Patch the run_game function to support custom agents."""
    import unstable.collector as collector_module
    from unstable._types import GameSpec, GameInformation, PlayerTrajectory
    from unstable.actor import VLLMActor
    import textarena as ta
    
    original_run_game = collector_module.run_game
    
    @ray.remote(num_cpus=0)
    def patched_run_game(game_spec: GameSpec, actor: VLLMActor, custom_agent_registry=None):
        """Patched version of run_game that supports custom agents."""
        game_information = GameInformation(
            game_idx=game_spec.game_idx, 
            eval_model_pid=game_spec.eval_model_pid, 
            eval_opponent_name=game_spec.eval_opponent_name
        )
        
        agents = {}
        for agent_spec in game_spec.agent_specs:
            if agent_spec.kind == "custom" and custom_agent_registry:
                # Handle custom agents
                agents[agent_spec.pid] = {
                    "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None,
                    "name": agent_spec.openrouter_name,  # Agent name
                    "kind": "custom",
                    "agent_registry": custom_agent_registry
                }
            else:
                # Use original logic for other agent types
                agents[agent_spec.pid] = {
                    "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None, 
                    "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
                    "kind": agent_spec.kind
                }
        
        # Initialize environment
        env = ta.make(game_spec.env_id, seed=game_spec.seed)
        env.reset()
        
        turn = 0
        while not env.done:
            current_pid = env.current_player
            agent_info = agents[current_pid]
            
            # Get observation
            observation = env.get_observation(current_pid)
            
            # Generate action based on agent type
            if agent_info["kind"] == "custom":
                # Use custom agent
                action = ray.get(agent_info["agent_registry"].call_agent.remote(
                    agent_info["name"], observation
                ))
            elif agent_info["kind"] == "checkpoint":
                # Use trained model
                action = ray.get(actor.generate.remote(
                    observation, agent_info["name"], 
                    game_spec.agent_specs[current_pid].prompt_template
                ))
            else:
                # Use OpenRouter (original behavior)
                action = ray.get(actor.generate.remote(
                    observation, agent_info["name"], 
                    game_spec.agent_specs[current_pid].prompt_template
                ))
            
            # Record trajectory if needed
            if agent_info["traj"] is not None:
                agent_info["traj"].add_step(observation, action, 0.0)
            
            # Step environment
            env.step(current_pid, action)
            turn += 1
        
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
    
    # Replace the original function
    collector_module.run_game = patched_run_game
    return patched_run_game
