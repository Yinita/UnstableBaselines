import ray, random
from unstable.utils import setup_logger
from unstable._types import AgentSpec, GameSpec, GameInformation

@ray.remote
class GameScheduler:
    def __init__(self, model_sampler, env_sampler, logging_dir: str):
        self.logger = setup_logger("game_scheduler", logging_dir)
        self.model_sampler = model_sampler
        self.env_sampler = env_sampler
        self._game_idx = 0
        self._running_jobs = {}

    def next_train_job(self):
        """
        生成下一个训练任务。始终返回有效的GameSpec或抛出异常。
        """
        try:
            self.logger.info("Creating next training job...")
            env_spec = self.env_sampler.sample(kind="train") # sample the env spec
            if env_spec is None:
                raise ValueError("Failed to sample environment specification for training")
                
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            if current_ckpt_lora_path is None:
                raise ValueError("Failed to get current checkpoint path")
                
            # build the game spec and agent specs
            pids = list(range(env_spec.num_players))
            random.shuffle(pids)
            agent_specs = []
            
            # 初始化运行任务记录
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": []}
            
            for i, pid in enumerate(pids):
                if i < env_spec.num_actors: # add current ckpt
                    self._running_jobs[self._game_idx]["models"].append({"uid": current_ckpt_uid, "pid": pid, "type": "model"})
                    agent_specs.append(AgentSpec(
                        pid=pid, 
                        kind="checkpoint", 
                        collect_data=True, 
                        lora_path=current_ckpt_lora_path, 
                        prompt_template=env_spec.prompt_template, 
                        action_extraction_fn=env_spec.action_extraction_fn
                    ))
                else: # sample opponent and add
                    opp_uid, kind, opp_lora_path, opp_openrouter_name = self.model_sampler.sample_opponent()
                    if kind is None:
                        raise ValueError(f"Failed to sample opponent for training job")
                        
                    agent_specs.append(AgentSpec(
                        pid=pid, 
                        kind=kind, 
                        lora_path=opp_lora_path, 
                        openrouter_name=opp_openrouter_name
                    ))
                    self._running_jobs[self._game_idx]["models"].append({"uid": opp_uid, "pid": pid, "type": "opponent"})
            
            # 生成游戏规格
            game_spec = GameSpec(
                game_idx=self._game_idx, 
                env_id=env_spec.env_id, 
                seed=self._game_idx, 
                agent_specs=agent_specs
            )
            
            self.logger.info(f"Created training job: game_idx={game_spec.game_idx}, "
                          f"env_id={game_spec.env_id}, seed={game_spec.seed}, "
                          f"num_agents={len(agent_specs)}")
            
            # 增加游戏索引
            self._game_idx += 1
            return game_spec
            
        except Exception as exc:
            self.logger.error(f"Exception in 'next_train_job': {exc}", exc_info=True)
            # 不要静默返回None，而是抛出异常
            raise ValueError(f"Failed to create training job: {exc}") from exc

    def next_eval_job(self):
        """
        生成下一个评估任务。始终返回有效的GameSpec或抛出异常。
        支持标准EvalEnvSpec和扩展的MixedPlayEvalEnvSpec。
        """
        try:
            self.logger.info("Creating next evaluation job...")
            env_spec = self.env_sampler.sample(kind="eval")
            if env_spec is None:
                raise ValueError("Failed to sample environment specification for evaluation")
                
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            if current_ckpt_lora_path is None:
                raise ValueError("Failed to get current checkpoint path")
                
            pids = list(range(env_spec.num_players))
            random.shuffle(pids)
            agent_specs = []
            
            # 检测是否为支持opponent_mapping的环境规格
            has_opponent_mapping = hasattr(env_spec, 'opponent_mapping') and hasattr(env_spec, 'get_opponent_for_position')
            self.logger.info(f"Environment spec {'supports' if has_opponent_mapping else 'does not support'} opponent mapping")
            
            # 记录所有使用的对手名称，用于评估结果记录
            used_opponent_names = []
            
            for i, pid in enumerate(pids):
                position = pid + 1  # 位置从1开始
                
                if i == 0:  
                    # 第一个位置始终用于评估的模型
                    self.logger.info(f"Setting position {position} (pid={pid}) as evaluation model")
                    agent_specs.append(AgentSpec(
                        pid=pid, 
                        kind="checkpoint", 
                        collect_data=True, 
                        lora_path=current_ckpt_lora_path, 
                        prompt_template=env_spec.prompt_template, 
                        action_extraction_fn=env_spec.action_extraction_fn
                    ))
                else:
                    # 其他位置使用对手
                    opponent_name = None
                    
                    # 如果支持opponent_mapping，则优先使用映射中的对手
                    if has_opponent_mapping:
                        opponent_name = env_spec.get_opponent_for_position(position)
                        if opponent_name:
                            self.logger.info(f"Using mapped opponent '{opponent_name}' for position {position} (pid={pid})")
                    
                    # 如果没有映射或映射为空，则使用默认固定对手
                    if not opponent_name:
                        opponent_name = env_spec.fixed_opponent
                        if not opponent_name:
                            raise ValueError(f"No opponent specified for position {position} in evaluation environment {env_spec.env_id}")
                        self.logger.info(f"Using default fixed opponent '{opponent_name}' for position {position} (pid={pid})")
                    
                    # 记录使用的对手名称
                    used_opponent_names.append(opponent_name)
                    
                    # 创建AgentSpec
                    agent_specs.append(AgentSpec(
                        pid=pid, 
                        kind="openrouter", 
                        lora_path=None, 
                        openrouter_name=opponent_name
                    ))
            
            # 生成游戏规格
            game_spec = GameSpec(
                game_idx=self._game_idx, 
                env_id=env_spec.env_id, 
                seed=self._game_idx, 
                agent_specs=agent_specs, 
                eval_model_pid=pids[0], 
                eval_opponent_name=','.join(used_opponent_names) if used_opponent_names else env_spec.fixed_opponent
            )
            
            self.logger.info(f"Created evaluation job: game_idx={game_spec.game_idx}, "
                          f"env_id={game_spec.env_id}, seed={game_spec.seed}, "
                          f"eval_model_pid={game_spec.eval_model_pid}, "
                          f"eval_opponent_name={game_spec.eval_opponent_name}")
            
            # 增加游戏索引
            self._game_idx += 1
            return game_spec
            
        except Exception as exc:
            self.logger.error(f"Exception in 'next_eval_job': {exc}", exc_info=True)
            # 不要静默返回None，而是抛出异常
            raise ValueError(f"Failed to create evaluation job: {exc}") from exc

    def update(self, game_info: GameInformation):
        job_info = self._running_jobs.pop(game_info.game_idx, None)
        if job_info is None: return # shouldn’t happen
        actor_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "model" if m["pid"] in game_info.final_rewards]
        opp_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "opponent" if m["pid"] in game_info.final_rewards]
        self.env_sampler.update(avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None), avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None)) # update environment sampling 
        self.model_sampler.update(game_info=game_info, job_info=job_info) # update model sampler

