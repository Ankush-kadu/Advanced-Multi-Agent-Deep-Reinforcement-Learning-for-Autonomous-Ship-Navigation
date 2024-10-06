from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import os
import yaml
from class_vessel import Vessel
from class_waves import Waves
import module_shared as sh
from utils import *
from gymnasium import spaces
import numpy as np
import os

class Multi_Agent_Env(MultiAgentEnv):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # print(f"Initializing environment with config: {config}")
        # print("Current working directory:", os.getcwd())
        self.input_file = self.config.get('input_file')
        # print(f"Looking for world file at: {self.world_file}")
        if not self.input_file:
            raise ValueError("Input_file file path must be specified in the config.")
        
        with open(self.input_file) as stream:
            self.input_data = yaml.safe_load(stream)

        self.action_space = spaces.Box(low=-np.radians(35), high=np.radians(35), shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-100, -np.pi, -100, -1], dtype=np.float64),
                                            high=np.array([100, np.pi, 100, 1], dtype=np.float64))
        
        self.num_agents = len(self.input_data['agents'])
        self._agent_ids = list(range(self.num_agents))
        self.max_steps = self.input_data["max_steps"]
        self.min_goal_dist = self.input_data["min_goal_dist"]
        self.max_goal_dist = self.input_data["max_goal_dist"]
        self.dt = self.input_data['time_step']
        self.agent_count = 0
        self.vessels = []
        print(f"self.input_data ={self.input_data} ")
        self.process_world_input(self.input_data)

        self.episode_count = 0
        self.count_step = 0
        self.terminateds = set()
        self.truncateds = set()
        self.agent_states = {}
        self.reset()

    def process_world_input(self, data=None):
        
        try:

            self.size = np.array(data['world_size'])

            sh.current_time = 0
            sh.dt = data['time_step']
            # self.num_agents = data['nagents']
            agent_count = 0
            
            for agent in data['agents'][0:self.num_agents]:
                # Appends the objects of class 'Vessel' to the list 'vessels'
                self.vessels.append(Vessel(vessel_data=agent, vessel_id=agent_count))
                agent_count += 1

            if data.get('waves') is not None:
                self.waves = Waves(data['waves'])
            
            if data.get('density') is not None:
                sh.rho = data['density']
            
            if data.get('gravity') is not None:
                sh.g = data['gravity']

        except yaml.YAMLError as exc:
            print(exc)
            exit()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.terminateds.clear()
        self.truncateds.clear()
        self.count_step = 0

        if self.episode_count % 10 == 0:
            os.makedirs('rl_trained_models/result_npy/PPO', exist_ok=True)
            np.save(f'rl_trained_models/result_npy/PPO/EP_{self.episode_count}', self.save_list)

        print(f"End of Episode {self.episode_count}")
        self.save_list = np.zeros((self.max_steps, self.num_agents, 15), dtype=np.float64)

        # self.agents = np.array([self.reset_agent(agent_id) for agent_id in self._agent_ids])
        observations_dict = {}
        info_dict = {}
        for id, agent in enumerate(self.vessels): 
            agent.inti_state()
            observations_dict[id], _, _, _, info_dict[id] = agent.pf_calculate()
            self.agent_states[id] = observations_dict[id]
        return observations_dict, info_dict
    
    def step(self, actions: dict):
        observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict = {}, {}, {}, {}, {}

        for agent_id, action in actions.items():
            self.vessels[agent_id].step(action) # = self.agent_state_update(agent_id, action)
            observation, reward, terminated, truncated, info  = self.vessels[agent_id].pf_calculate()
            self.agent_states[agent_id] = observation

            if terminated:
                self.terminateds.add(agent_id)
            if truncated:
                self.truncateds.add(agent_id)

            observation_dict[agent_id] = observation
            reward_dict[agent_id] = reward
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = info

        terminated_dict["__all__"] = len(self.terminateds) == self.num_agents
        truncated_dict["__all__"] = self.count_step >= self.max_steps - 1

        self.count_step += 1

        return observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict
