import numpy as np
import asyncio
import websockets
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  

clients = set()

class ROS2WebSocketBridge(Node):
    def __init__(self):
        super().__init__('ros2_websocket_bridge')
        self.subscription = self.create_subscription(String, '/your_ros_topic', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

    async def listener_callback(self, msg):
        # Forward the received ROS message to all connected WebSocket clients
        message_data = str(msg.data)  # Adjust based on your message type
        if clients:  # Check if there are any connected WebSocket clients
            await asyncio.wait([client.send(message_data) for client in clients])

async def register_websocket(websocket, path):
    # Register a new WebSocket connection
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def main():
    # Initialize the ROS 2 node
    rclpy.init(args=None)
    ros2_node = ROS2WebSocketBridge()

    # Start the WebSocket server
    start_server = websockets.serve(register_websocket, 'localhost', 8765)

    # Run both ROS 2 and WebSocket server event loops
    await asyncio.gather(
        rclpy.spin(ros2_node, executor=asyncio.get_event_loop()),
        start_server,
    )

if __name__ == '__main__':
    asyncio.run(main())
