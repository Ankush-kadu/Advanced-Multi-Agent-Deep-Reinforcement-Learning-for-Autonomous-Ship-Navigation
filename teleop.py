import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32,Int32MultiArray,Float32 # Use Int32 message type  (Publishing Normalized values as Float32)
import serial
import numpy as np

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.propeller_publisher = self.create_publisher(Float32, 'matsya_00/propeller_cmd', 10)
        self.rudder_publisher = self.create_publisher(Float32, 'matsya_00/rudder_cmd', 10)
        self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        self.ser.reset_input_buffer()
        self.rfpublisher = self.create_publisher(Int32MultiArray, '/RFval', 10)
        timer_period = 0.008  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer_pub = self.create_timer(1, self.pub_callback)

        self.propmsg = Float32()
        self.rudmsg = Float32()

        self.prop_list = []
        self.rud_list = []

    def pub_callback(self):
        if len(self.rud_list) == 0:
            return
        rud = Float32()
        prop = Float32()
        rud.data = np.mean(np.array(self.rud_list))
        prop.data = np.mean(np.array(self.prop_list))

        self.propeller_publisher.publish(prop)
        self.rudder_publisher.publish(rud)
        print(rud.data, prop.data)
        self.prop_list.clear()
        self.rud_list.clear()


    def timer_callback(self):
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').rstrip()
            val = line.split(',')
            msg = Int32MultiArray()
            propmsg = Float32()
            rudmsg = Float32()
            arr = [int(i) for i in val]
            msg.data = arr
            rudmsg.data = float(self.num_to_range(arr[3],1300,1700,-35*np.pi/180,35*np.pi/180)) # Rudder angle between -35 and 35
            propmsg.data = float(self.num_to_range(arr[1],1500,1700,0,20)) # Normalized proppeller speed between 0 and 5
            self.propmsg = propmsg
            self.rudmsg = rudmsg

            self.prop_list.append(propmsg.data)
            self.rud_list.append(rudmsg.data)
            # self.rfpublisher.publish(msg)
        else:pass


    def num_to_range(self,num, inMin, inMax, outMin, outMax):
        if num < inMin:
            return outMin
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))



def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopNode()
    rclpy.spin(teleop_node)
    teleop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
