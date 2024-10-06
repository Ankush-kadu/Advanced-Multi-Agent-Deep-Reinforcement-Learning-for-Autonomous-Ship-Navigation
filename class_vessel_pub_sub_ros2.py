import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, Image, LaserScan, NavSatFix, NavSatStatus
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion, Twist, PoseWithCovarianceStamped
from interfaces.msg import Actuator
import module_shared as sh
import module_kinematics as kin

class Vessel_Pub_Sub():
    sensors = []
    actuators = []
    vessel_id = None
    vessel_name = None
    topic_prefix = None
    n_c = 0         # Test turning circle -> 40/3
    delta_c = 0     # Test turning circle -> 35 * np.pi / 180

    def __init__(self, vessel_id):
        self.vessel_id = vessel_id
        self.vessel_name = sh.world.vessels[self.vessel_id].name
        self.topic_prefix = f'{self.vessel_name}_{self.vessel_id:02d}'

        self.sensors = []
        for sensor in sh.world.vessels[self.vessel_id].sensors:
            self.register_sensor(sensor)
        self.register_actuators()
    
    def register_actuators(self):
        self.actuators.append(sh.world.node.create_subscription(Actuator, f'{self.topic_prefix}/actuator_cmd', self.actuator_callback, 1))

    def register_sensor(self, sensor):
        rate = sensor['publish_rate']

        sensor['id'] = len(self.sensors)
        self.sensors.append(sensor)

        if sensor['sensor_type'] == 'IMU':
            loc = np.array(sensor['sensor_location'])
            orient = np.array(sensor['sensor_orientation'])
            # self.sensors[-1]['pub'] = sh.world.node.create_publisher(Imu, f'{self.topic_prefix}/imu_{sensor["id"]:02d}', 10)
            self.sensors[-1]['pub'] = sh.world.node.create_publisher(Imu, sensor['topic'], 10)
            self.sensors[-1]['timer'] = sh.world.node.create_timer(1/rate, lambda: self.publish_imu(id=sensor['id'], loc=loc, orient=orient))

        elif sensor['sensor_type'] == 'GPS':
            loc = np.array(sensor['sensor_location'])
            # self.sensors[-1]['pub'] = sh.world.node.create_publisher(NavSatFix, f'{self.topic_prefix}/gps_{sensor["id"]:02d}', 10)
            self.sensors[-1]['pub'] = sh.world.node.create_publisher(NavSatFix, sensor['topic'], 10)
            self.sensors[-1]['timer'] = sh.world.node.create_timer(1/rate, lambda: self.publish_gps(id=sensor['id'], loc=loc))
        
        elif sensor['sensor_type'] == 'UWB':
            loc = np.array(sensor['sensor_location'])
            # self.sensors[-1]['pub'] = sh.world.node.create_publisher(PoseWithCovarianceStamped, f'{self.topic_prefix}/uwb_{sensor["id"]:02d}', 10)
            self.sensors[-1]['pub'] = sh.world.node.create_publisher(PoseWithCovarianceStamped, sensor['topic'], 10)
            self.sensors[-1]['timer'] = sh.world.node.create_timer(1/rate, lambda: self.publish_uwb(id=sensor['id'], loc=loc))
        
        elif sensor['sensor_type'] == 'encoders':
            # self.sensors[-1]['pub'] = sh.world.node.create_publisher(Actuator, f'{self.topic_prefix}/encoders_{sensor["id"]:02d}', 10)
            self.sensors[-1]['pub'] = sh.world.node.create_publisher(Actuator, sensor['topic'], 10)
            self.sensors[-1]['timer'] = sh.world.node.create_timer(1/rate, lambda: self.publish_actuator(id=sensor['id']))
        
        else:
            raise ValueError("Specified sensor type is unknown")
        
    def publish_imu(self, id, loc, orient):
        current_time = sh.world.node.get_clock().now()

        # Create odometry message
        imu = Imu()
        imu.header.stamp = current_time.to_msg()
        imu.header.frame_id = self.topic_prefix + f'_imu_{id:02d}_frame'

        eul_rms = np.array([1, 1, 1]) * 1e-3 # * np.pi / 180
        eul_cov = np.diag(eul_rms ** 2)

        ang_vel_rms = np.array([1, 1, 1], dtype=np.float64) * 1e-2
        ang_vel_cov = np.diag(ang_vel_rms ** 2)

        lin_acc_rms = np.array([1, 1, 1], dtype=np.float64) * 1e-2
        lin_acc_cov = np.diag(lin_acc_rms ** 2)

        imu.orientation_covariance = eul_cov.flatten()
        imu.angular_velocity_covariance = ang_vel_cov.flatten()
        imu.linear_acceleration_covariance = lin_acc_cov.flatten()

        state = sh.world.vessels[self.vessel_id].current_state
        state_der = sh.world.vessels[self.vessel_id].current_state_der
        
        quat = state[9:13]
        omg_bcs = state[3:6]
        v_bcs = state[0:3]        
        alpha = state_der[3:6]

        # q_sensor = kin.quat_multiply(quat, orient)
        q_sensor = kin.rotm_to_quat(kin.quat_to_rotm(quat) @ kin.quat_to_rotm(orient))
        q_sensor = kin.eul_to_quat(kin.quat_to_eul(q_sensor) + np.random.multivariate_normal(np.zeros(3), eul_cov))
        
        acc_bcs = state_der[0:3] + np.cross(omg_bcs, v_bcs)
        acc_s_bcs = acc_bcs + np.cross(alpha, loc) + np.cross(omg_bcs, np.cross(omg_bcs, loc))
        acc_sensor = kin.quat_to_rotm(orient).T @ acc_s_bcs
        acc_sensor = acc_sensor + kin.quat_to_rotm(q_sensor).T @ np.array([0, 0, -sh.g])
        acc_sensor = acc_sensor + np.random.multivariate_normal(np.zeros(3), lin_acc_cov)

        omg_sensor = kin.quat_to_rotm(orient).T @ omg_bcs
        omg_sensor = omg_sensor + np.random.multivariate_normal(np.zeros(3), ang_vel_cov)
                
        imu.orientation = Quaternion(x=q_sensor[1], y=q_sensor[2], z=q_sensor[3], w=q_sensor[0])
        imu.angular_velocity = Vector3(x=omg_sensor[0], y=omg_sensor[1], z=omg_sensor[2])
        imu.linear_acceleration = Vector3(x=acc_sensor[0], y=acc_sensor[1], z=acc_sensor[2])

        self.sensors[id]['pub'].publish(imu)

    def publish_gps(self, id, loc):
        current_time = sh.world.node.get_clock().now()

        gps_rms = np.array([3, 3, 3], dtype=np.float64)
        gps_cov = np.diag(gps_rms ** 2)

        state = sh.world.vessels[self.vessel_id].current_state
        llh0 = np.array([12.993661, 80.239402, 0])
        ned = state[6:9] + kin.quat_to_rotm(state[9:13]) @ loc 
        ned = ned + np.random.multivariate_normal(np.zeros(3), gps_cov)
        llh = kin.ned_to_llh(ned, llh0)

        # Create NavSatFix message
        gps = NavSatFix()
        gps.header.stamp = current_time.to_msg()
        gps.header.frame_id = 'ECEF'
        gps.status = NavSatStatus(status=0, service=1)
        gps.latitude = llh[0]
        gps.longitude = llh[1]
        gps.altitude = llh[2]
        gps.position_covariance = gps_cov.flatten()

        self.sensors[id]['pub'].publish(gps)
    
    def publish_uwb(self, id, loc):
        current_time = sh.world.node.get_clock().now()

        uwb_rms = np.array([1, 1, 1], dtype=np.float64)
        uwb_cov = np.diag(uwb_rms ** 2)
        
        uwb_cov_full = -np.eye(6)               # Specify -1 for orientation covariance
        uwb_cov_full[0:3][:, 0:3] = uwb_cov

        state = sh.world.vessels[self.vessel_id].current_state
        ned = state[6:9]
        r_sen = ned + kin.quat_to_rotm(state[9:13]) @ loc
        r_sen = r_sen + np.random.multivariate_normal(np.zeros(3), uwb_cov)

        # Create UWB PointStamped message
        uwb = PoseWithCovarianceStamped()
        uwb.header.stamp = current_time.to_msg()
        uwb.header.frame_id = 'NED'
        uwb.pose.pose.position = Point(x=r_sen[0], y=r_sen[1], z=r_sen[2])
        uwb.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        uwb.pose.covariance = uwb_cov_full.flatten()
        
        self.sensors[id]['pub'].publish(uwb)
    
    def publish_actuator(self, id):
        current_time = sh.world.node.get_clock().now()

        state = sh.world.vessels[self.vessel_id].current_state
        
        encoders_rms = np.array([1e-1 * np.pi / 180, 1e-1 / 60])
        encoders_cov = np.diag(encoders_rms ** 2)
        
        act = Actuator()
        act.header.stamp = current_time.to_msg()
        act.rudder = state[13] * 180.0 / np.pi
        act.propeller = state[14] * 60.0
        act.covariance = encoders_cov.flatten()

        self.sensors[id]['pub'].publish(act)

    def actuator_callback(self, msg):
        self.delta_c = msg.rudder * np.pi / 180.0
        self.n_c = msg.propeller / 60.0
