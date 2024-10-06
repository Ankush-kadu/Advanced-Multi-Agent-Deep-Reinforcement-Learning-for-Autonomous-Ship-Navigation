#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('/workspaces/makara/ros2_ws/install/mav_simulator/lib/python3.10/site-packages/mav_simulator/')

from class_world import World
import module_shared as sh
import asyncio
import json
import websockets
import threading
import rclpy
from class_world_node_ros2 import World_Node
import module_kinematics as kin

def ros_thread():
    rclpy.spin(sh.world.node)
    rclpy.shutdown()

def main():
    # Creates an object of class 'World'
    sh.world = World('/workspaces/makara/inputs/inputs.yml')

    # rclpy.init()
    # world_node = World_Node(world_rate=1/sh.dt)
    # sh.world.node = world_node
    # sh.world.start_vessel_ros_nodes()
    # rclpy.spin(world_node)
    # rclpy.shutdown()

    rclpy.init()
    world_node = World_Node(world_rate=1/sh.dt)
    sh.world.node = world_node
    sh.world.start_vessel_ros_nodes()

    # Run ROS on a separate thread
    ros_thread_instance = threading.Thread(target=ros_thread)
    ros_thread_instance.start()

    # Run asyncio on a separate thread
    async def handler(websocket):
        while True:
            for vessel in sh.world.vessels:
                state = vessel.current_state
                data = {}
                data['vessel'] = vessel.vessel_node.topic_prefix
                data['rudder'] = state[13] * 180 / np.pi
                data['propeller'] = state[14] * 60
                
                await websocket.send(json.dumps(data))
            sleep_time = np.max(np.array([1/sh.world.node.rate, 1/30.0]))
            await asyncio.sleep(sleep_time)

    start_server = websockets.serve(handler, "0.0.0.0", 9002)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    main()