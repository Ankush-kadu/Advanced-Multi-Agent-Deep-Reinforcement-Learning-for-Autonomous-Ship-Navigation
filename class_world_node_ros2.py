import rclpy
from rclpy.node import Node
import module_shared as sh

class World_Node(Node):
    rate = None
    def __init__(self, world_rate=100):
        super().__init__('world')
        self.rate = world_rate
        self.create_timer(1/self.rate, callback=sh.world.step)