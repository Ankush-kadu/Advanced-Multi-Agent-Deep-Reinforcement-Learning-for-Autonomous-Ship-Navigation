import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, Image, LaserScan, NavSatFix
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion, Twist
import module_shared as sh

class World_Node(Node):
    def __init__(self, world_rate=10):
        super().__init__('world')
        self.create_timer(1/world_rate, callback=sh.world.step)


class Vessel_Pub_Sub():
    publishers = []
    timers = []
    odom_frame_id = None
    base_frame_id = None
    vessel_id = None
    vessel_name = None
    node_name = None
    odom_rate = 10
    n_c = 0
    delta_c = 0
    
    def __init__(self, vessel_id, odom_rate=10):
        self.vessel_id = vessel_id
        self.vessel_name = sh.world.vessels[self.vessel_id].name
        self.node_name = f'{self.vessel_name}_{self.vessel_id:02d}'
        # super().__init__(self.node_name)
        self.odom_rate = odom_rate
        self.create_odometry()
        self.create_rudder_subscriber()
        self.create_prop_subscriber()

    def guidance(self):
        pass

    def controller(self):
        pass
    
    def create_imu(self):
        self.publishers.append(sh.world.node.create_publisher(Imu, f'{self.node_name}/imu', 10))
        self.timers.append(sh.world.node.create_timer(1/self.odom_rate, self.publish_odometry))
    
    def create_odometry(self):
        self.publishers.append(sh.world.node.create_publisher(Odometry, f'{self.node_name}/odometry', 10))
        self.timers.append(sh.world.node.create_timer(1/self.odom_rate, self.publish_odometry))
        self.odom_frame_id = 'odom'
        self.base_frame_id = f'{self.node_name}_base_link'

    def create_prop_subscriber(self):
        sh.world.node.create_subscription(
            Float32,
            f'{self.node_name}/propeller_cmd',
            self.propeller_callback,
            1
        )

    def create_rudder_subscriber(self):
        sh.world.node.create_subscription(
            Float32,
            f'{self.node_name}/rudder_cmd',
            self.rudder_callback,
            1
        )

    def publish_odometry(self):
        current_time = sh.world.node.get_clock().now()

        # Create odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id

        state = sh.world.vessels[self.vessel_id].current_state

        # Set the position
        odom.pose.pose.position = Point(x=state[6], y=state[7], z=state[8])
        odom.pose.pose.orientation = Quaternion(x=state[10], y=state[11], z=state[12], w=state[9])

        # Set the velocity
        odom.twist.twist.linear = Vector3(x=state[0], y=state[1], z=state[2])
        odom.twist.twist.angular = Vector3(x=state[3], y=state[4], z=state[5])

        # Publish the message
        self.publishers[0].publish(odom)

    def propeller_callback(self, msg):
        self.n_c = msg.data

    def rudder_callback(self, msg):
        self.delta_c = np.radians(msg.data)

