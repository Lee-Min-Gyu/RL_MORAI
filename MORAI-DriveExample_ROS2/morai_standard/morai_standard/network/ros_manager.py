#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from morai_ros2_msgs.msg import EgoVehicleStatus, ObjectStatusList, CtrlCmd, GetTrafficLightStatus, SetTrafficLight

from morai_standard.autonomous_driving.vehicle_state import VehicleState
from morai_standard.autonomous_driving.perception.object_info import ObjectInfo
from morai_standard.autonomous_driving.config.config import Config


class RosManager(Node):
    def __init__(self, autonomous_driving):
        super().__init__('morai_standard')
        self.autonomous_driving = autonomous_driving

        config = Config()

        self.traffic_light_control = config["map"]["traffic_light_control"]
        self.global_path = self.convert_to_ros_path(config["map"]["path"], '/map')

        self.sampling_rate = config["common"]["sampling_rate"]
        self.count = 0

        self.vehicle_state = VehicleState()
        self.object_info_list = []
        self.traffic_light = []

        self.is_status = False
        self.is_object_info = False
        self.is_traffic_light = False

        # TF Broadcaster
        self.br = tf2_ros.TransformBroadcaster(self)

        # publisher
        self.global_path_pub = self.create_publisher(Path, '/global_path', 5)
        self.local_path_pub = self.create_publisher(Path, '/local_path', 5)
        self.ctrl_pub = self.create_publisher(CtrlCmd, '/ctrl_cmd_0', 5)
        self.traffic_light_pub = self.create_publisher(SetTrafficLight, '/traffic_light_control', 5)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 5)

        # subscriber
        self.create_subscription(EgoVehicleStatus, "/ego_vehicle_status", self.vehicle_status_callback, 5)
        self.create_subscription(ObjectStatusList, "/object_status", self.object_info_callback, 5)
        self.create_subscription(GetTrafficLightStatus, "/traffic_light_status", self.traffic_light_callback, 5)
        
        self.timer = self.create_timer(1.0 / self.sampling_rate, self.execute)

    def execute(self):
        if self.is_status and self.is_object_info:
            control_input, local_path = self.autonomous_driving.execute(
                self.vehicle_state, self.object_info_list, self.traffic_light
            )
            self._send_data(control_input, local_path)

    def _send_data(self, control_input, local_path):
        self.ctrl_pub.publish(CtrlCmd(**control_input.__dict__))
        self.local_path_pub.publish(self.convert_to_ros_path(local_path, 'map'))
        self.odom_pub.publish(self.convert_to_odometry(self.vehicle_state))

        if self.count == self.sampling_rate:
            self.global_path_pub.publish(self.global_path)
            self.count = 0
        self.count += 1

    @staticmethod
    def convert_to_ros_path(path, frame_id):
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        for point in path:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position = Point(x=point.x, y=point.y, z=0.0)
            pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ros_path.poses.append(pose_stamped)

        return ros_path

    def convert_to_odometry(self, vehicle_state):
        odometry = Odometry()
        odometry.header.frame_id = 'map'
        odometry.child_frame_id = 'gps'

        q = self.euler_to_quaternion(0.0, 0.0, vehicle_state.yaw)

        odometry.pose.pose.position = Point(x=vehicle_state.position.x, y=vehicle_state.position.y, z=0.0)
        odometry.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return odometry

    def vehicle_status_callback(self, data):
        self.vehicle_state = VehicleState(data.position.x, data.position.y, np.deg2rad(data.heading), data.velocity.x)
        # TF Broadcast
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'gps'
        t.transform.translation.x = float(self.vehicle_state.position.x)
        t.transform.translation.y = float(self.vehicle_state.position.y)
        t.transform.translation.z = 0.0
        
        q = self.euler_to_quaternion(0.0, 0.0, self.vehicle_state.yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.br.sendTransform(t)
        self.is_status = True

    def object_info_callback(self, data):
        self.object_info_list = [
            ObjectInfo(data.position.x, data.position.y, data.velocity.x, data.type)
            for data in data.npc_list + data.obstacle_list + data.pedestrian_list
        ]
        self.is_object_info = True

    def traffic_light_callback(self, data):
        # traffic_control (차량 신호 Green Light(16) 변경)
        if self.traffic_light_control:
            traffic_light_status = 16
            set_traffic_light = SetTrafficLight(
                traffic_light_index=data.traffic_light_index,
                traffic_light_status=traffic_light_status
            )
            self.traffic_light_pub.publish(set_traffic_light)
        else:
            traffic_light_status = data.traffic_light_status

        self.traffic_light = [data.traffic_light_index, traffic_light_status]
        self.is_traffic_light = True

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]
    
def run_node(autonomous_driving):
    rclpy.init()
    ros_manager = RosManager(autonomous_driving)
    rclpy.spin(ros_manager)