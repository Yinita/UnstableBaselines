import ray, random
from unstable.types import AgentSpec, GameSpec, GameInformation

@ray.remote
class GameScheduler:
    def __init__(self, model_sampler, env_sampler):
        self.model_sampler = model_sampler
        self.env_sampler = env_sampler
        self._game_idx = 0
        self._running_jobs = {}

    def next_train_job(self):
        env_spec = self.env_sampler.sample(kind="train") # sample the env spec
        current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint

        # build the game spec and agent specs
        pids = list(range(env_spec.num_players))
        random.shuffle(pids); agent_specs = []
        self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": []}
        for i, pid in enumerate(pids):
            if i < env_spec.num_actors: # add current ckpt
                self._running_jobs[self._game_idx]["models"].append({"uid": current_ckpt_uid, "pid": pid, "type": "model"})
                agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn))
            else: # sample opponent and add
                opp_uid, kind, opp_lora_path, opp_openrouter_name = ray.get(self.model_sampler.sample.remote()) 
                agent_specs.append(AgentSpec(pid=pid, kind=kind, lora_path=opp_lora_path, openrouter_name=opp_openrouter_name)) # TODO might have to adjust what is passed
                self._running_jobs[self._game_idx]["models"].append({"uid": opp_uid, "pid": pid, "type": "opponent"})

        # populate GameSpec
        game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs)
        self._game_idx += 1
        return game_spec

    def next_eval_job(self):
        # make sure to set collect_data=False for eval games TODO
        raise NotImplementedError
        # env = self.env_sampler.sample("eval")
        # return env, ray.get(self.model_sampler.get_current_ckpt.remote())

    def update(self, game_info: GameInformation):
        job_info = self._running_jobs.pop(game_info.game_idx, None)
        if job_info is None: return # shouldn’t happen
        actor_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "model" if m["pid"] in game_info.final_rewards]
        opp_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "opponent" if m["pid"] in game_info.final_rewards]
        self.env_sampler.update(avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None), avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None)) # update environment sampling 
        self.model_sampler.update(game_info=game_info, job_info=job_info) # update model sampler




