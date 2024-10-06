#!/usr/bin/env python3

import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.ppo import PPOConfig
from Multi_Agent_Modified import Multi_Agent_Env
from gymnasium import spaces
import numpy as np
import os

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_0"

params = {
    "input_file": '/workspaces/makara/inputs/inputs.yml',  
    "max_steps": 200,
}

observation_space = spaces.Box(low=np.array([-100, -np.pi, -np.inf, -1], dtype=np.float64), 
                       high=np.array([100, np.pi, np.inf, 1], dtype=np.float64))
# action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64)
action_space = spaces.Box(low=-np.radians(35), high=np.radians(35), shape=(1,), dtype=np.float64)

config = PPOConfig()
config.environment(env=Multi_Agent_Env, env_config=params)
config.framework(framework="torch")
config.resources(num_gpus=0)
config.env_runners(num_env_runners=10)

config.model = {
    "fcnet_hiddens": [128, 128],
    "fcnet_activation": "tanh",
}

config.training(
    clip_param=0.2,
    lambda_=0.95,
    entropy_coeff=0.01
)

config.lr = 0.001
config.gamma = 0.95
config.timesteps_per_iteration = 50 * params['max_steps']

config.multi_agent(
    policies={f"policy_0": (None, observation_space, action_space, {})},
    policy_mapping_fn=policy_mapping_fn,
    policies_to_train=[f"policy_0"]
)

results_dir = os.path.join(os.getcwd(), "rl_trained_models")
os.makedirs(results_dir, exist_ok=True)

training_iteration = 150
tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"training_iteration": training_iteration}, 
                             storage_path=results_dir),  
    param_space=config.to_dict()
)

tuner.fit()
ray.shutdown()

