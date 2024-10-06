#!/usr/bin/env python3

import numpy as np
import yaml
from class_vessel import Vessel
from class_waves import Waves
import module_shared as sh
from class_vessel_pub_sub_ros2 import Vessel_Pub_Sub

class World():

    terminate = False                   # variable to note whether world is terminated
    vessels = []                        # list to store objects of class 'Vessel'
    nvessels = 0                        # number of vessels in the world
    size = np.zeros(3)                  # Initialization of Size of the world (X-Y-Z)
    node = None

    waves = None

    def __init__(self, world_file=None):
        if world_file is not None:
            with open(world_file) as stream:
                world_data = yaml.safe_load(stream)  # content from world_file.yml stored in world_data
            self.process_world_input(world_data)       # parse the world data dictionary to create dictionaries

    def start_vessel_ros_nodes(self):

        for vessel in self.vessels:
            vessel.vessel_node = Vessel_Pub_Sub(vessel_id=vessel.vessel_id)
    
    def process_world_input(self, data=None):
        
        try:
            self.size = np.array(data['world_size'])
            sh.current_time = 0
            sh.dt = data['time_step']
            self.nvessels = data['nagents']
            agent_count = 0
            
            for agent in data['agents'][0:self.nvessels]:
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

    def step(self):
        for vessel in self.vessels:
            vessel.step()
        sh.current_time += sh.dt
