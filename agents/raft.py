#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
RAFT-Stereo based autonomous agent for lunar navigation with ROS integration.
This agent demonstrates real-time stereo depth estimation using RAFT-Stereo,
publishes sensor data to ROS topics, and provides manual control capabilities.
"""

import sys
import os
import threading
import argparse
import time

# Add the agents directory to Python path to ensure core module imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Standard libraries
import numpy as np
import random
from math import radians

# Computer Vision
import cv2 as cv
import open3d as o3d

# PyTorch
import torch

# CARLA
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

# RAFT-Stereo dependencies
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

# Scientific computing
from scipy.spatial.transform import Rotation as R

# Control
from pynput import keyboard

# ROS dependencies
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField, Imu, Image, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
import geometry_msgs.msg
from cv_bridge import CvBridge
import std_msgs.msg


def get_entry_point():
    """Entry point for the leaderboard to instantiate the agent"""
    return 'RAFTStereoAgent'


class RAFTStereoAgent(AutonomousAgent):
    """
    RAFT-Stereo based autonomous agent for lunar navigation.
    
    Features:
    - Real-time stereo depth estimation using RAFT-Stereo
    - ROS integration for data publishing
    - Point cloud generation and filtering
    - Manual control via keyboard
    - Sensor data visualization
    """

    def setup(self, path_to_conf_file):
        """Initialize the agent with RAFT-Stereo model and ROS publishers"""
        
        # Initialize control variables
        self.current_v = 0
        self.current_w = 0
        self.frame = 0
        
        # Setup keyboard listener for manual control
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        # Initialize image enhancement
        self.clhae = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        
        # Initialize RAFT-Stereo model
        self._setup_raft_model()
        
        # Initialize ROS publishers and services
        self._setup_ros_publishers()
        
        # Initialize coordinate frame parameters
        self._setup_coordinate_frames()

    def _setup_raft_model(self):
        """Configure and load the RAFT-Stereo model"""
        class Args:
            restore_ckpt = "models/raftstereo-realtime.pth"
            mixed_precision = True
            corr_implementation = "reg"  # Use CPU implementation
            corr_levels = 4
            corr_radius = 4
            n_downsample = 3
            context_norm = 'instance'
            slow_fast_gru = True
            n_gru_layers = 2
            hidden_dims = [128, 128, 128]
            shared_backbone = True
            valid_iters = 10 

        self.args = Args()
        
        # Create and load model
        self.raft_model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        self.raft_model.load_state_dict(torch.load(self.args.restore_ckpt), strict=False)
        self.raft_model = self.raft_model.module
        self.raft_model.to('cuda')
        self.raft_model.eval()
        
        # Store utilities
        self.InputPadder = InputPadder
        self.cx = 170  # Camera center x
        self.cy = 200  # Camera center y

    def _setup_ros_publishers(self):
        """Initialize ROS node and publishers"""
        rospy.init_node('raft_stereo_agent', anonymous=True)
        
        # Point cloud publisher
        self.pcd_publisher = rospy.Publisher('/camera/point_cloud', PointCloud2, queue_size=10)
        
        # IMU publisher
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=50)
        
        # Stereo camera publishers
        self.left_pub = rospy.Publisher("camera/left/image_raw", Image, queue_size=10)
        self.right_pub = rospy.Publisher("camera/right/image_raw", Image, queue_size=10)
        self.left_info_pub = rospy.Publisher("camera/left/camera_info", CameraInfo, queue_size=10)
        self.right_info_pub = rospy.Publisher("camera/right/camera_info", CameraInfo, queue_size=10)
        
        # Enhanced image publishers
        self.left_eq_pub = rospy.Publisher("camera/left/image_equalized", Image, queue_size=10)
        self.right_eq_pub = rospy.Publisher("camera/right/image_equalized", Image, queue_size=10)
        
        # Depth map publisher
        self.depth_pub = rospy.Publisher("camera/depth/image_raw", Image, queue_size=10)
        
        # Pose publisher
        self.pose_pub = rospy.Publisher("robot/pose", PoseStamped, queue_size=10)
        
        # Start IMU publishing thread
        imu_thread = threading.Thread(target=self.imu_publisher, daemon=True)
        imu_thread.start()
        
        # Initialize point cloud data
        self.pcd = o3d.geometry.PointCloud()
        self.rate = rospy.Rate(30)

    def _setup_coordinate_frames(self):
        """Initialize coordinate frame transformations and TF broadcasting"""
        # Prepare message headers
        self.header = rospy.Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = "base_link"
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Get initial robot position and publish static transform
        initial_transform = self.get_initial_position()
        translation = np.array([
            initial_transform.location.x, 
            initial_transform.location.y, 
            initial_transform.location.z
        ])
        
        rpy = np.array([
            initial_transform.rotation.roll, 
            initial_transform.rotation.pitch, 
            initial_transform.rotation.yaw
        ])
        rot = R.from_euler('xyz', rpy)
        rotation = rot.as_quat()
        
        self.broadcast_static_tf(translation, rotation)
        self.timestamp = rospy.Time.now()

    def broadcast_static_tf(self, translation, rotation):
        """Publish static TF transform from world to base_link"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "base_link"
        
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        
        self.tf_broadcaster.sendTransform(t)

    def use_fiducials(self):
        """Specify whether agent uses fiducials for localization"""
        return False

    def sensors(self):
        """Configure sensor parameters for stereo cameras"""
        sensor_config = {
            'camera_active': False, 
            'light_intensity': 0, 
            'width': '1280', 
            'height': '720'
        }
        
        stereo_config = {
            'camera_active': True, 
            'light_intensity': 2.0, 
            'width': '640', 
            'height': '480'
        }
        
        return {
            carla.SensorPosition.Front: sensor_config,
            carla.SensorPosition.FrontLeft: {**stereo_config, 'use_semantic': True},
            carla.SensorPosition.FrontRight: stereo_config,
            carla.SensorPosition.Left: sensor_config,
            carla.SensorPosition.Right: sensor_config,
            carla.SensorPosition.BackLeft: sensor_config,
            carla.SensorPosition.BackRight: sensor_config,
            carla.SensorPosition.Back: sensor_config,
        }

    def compute_disparity(self, raft_model, left_image, right_image, iters):
        """Compute disparity using RAFT-Stereo for stereo image pair"""
        # Convert to PyTorch tensors
        left_torch = torch.from_numpy(left_image).permute(2, 0, 1).float()
        right_torch = torch.from_numpy(right_image).permute(2, 0, 1).float()
        
        # Move to GPU and add batch dimension
        left_torch = left_torch[None].cuda()
        right_torch = right_torch[None].cuda()
        
        # Pad images to required dimensions
        padder = self.InputPadder(left_torch.shape, divis_by=32)
        left_torch, right_torch = padder.pad(left_torch, right_torch)
        
        # Forward pass through RAFT-Stereo
        with torch.no_grad():
            _, flow_up = raft_model(left_torch, right_torch, iters=iters, test_mode=True)
        
        # Unpad and convert to numpy
        flow_up = padder.unpad(flow_up).squeeze(0).cpu().numpy()
        
        # Convert flow to disparity and clamp negative values
        disparity = -flow_up[0]
        disparity = np.where(disparity < 0, 0, disparity)
        
        return disparity

    def get_camera_info(self):
        """Generate camera info message for stereo cameras"""
        camera_info = CameraInfo()
        camera_info.width = 640
        camera_info.height = 480
        camera_info.distortion_model = "plumb_bob"
        camera_info.K = [458, 0, 320, 0, 458, 240, 0, 0, 1]
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        camera_info.P = [458, 0, 320, -458*0.162, 0, 458, 240, 0, 0, 0, 1, 0]
        return camera_info

    def numpy_to_ros_image(self, np_img, encoding="bgr8"):
        """Convert NumPy image to ROS Image message"""
        ros_img = Image()
        ros_img.header = std_msgs.msg.Header()
        ros_img.header.stamp = rospy.Time.now()
        ros_img.height = np_img.shape[0]
        ros_img.width = np_img.shape[1]
        ros_img.encoding = encoding
        ros_img.is_bigendian = 0
        ros_img.step = np_img.shape[1] * (3 if encoding == "bgr8" else 1)
        ros_img.data = np_img.tobytes()
        return ros_img

    def publish_stereo_images(self, left_image, right_image):
        """Publish stereo images and camera info to ROS topics"""
        # Convert grayscale to BGR if necessary
        if len(left_image.shape) == 2 or left_image.shape[2] == 1:
            left_bgr = cv.cvtColor(left_image, cv.COLOR_GRAY2BGR)
        else:
            left_bgr = left_image
            
        if len(right_image.shape) == 2 or right_image.shape[2] == 1:
            right_bgr = cv.cvtColor(right_image, cv.COLOR_GRAY2BGR)
        else:
            right_bgr = right_image

        # Create and publish image messages
        timestamp = rospy.Time.now()
        
        left_msg = self.numpy_to_ros_image(left_bgr, encoding="bgr8")
        right_msg = self.numpy_to_ros_image(right_bgr, encoding="bgr8")
        
        left_msg.header.frame_id = "camera_left"
        right_msg.header.frame_id = "camera_right"
        left_msg.header.stamp = timestamp
        right_msg.header.stamp = timestamp
        
        # Create and publish camera info messages
        left_info = self.get_camera_info()
        right_info = self.get_camera_info()
        
        left_info.header.frame_id = "camera_left"
        right_info.header.frame_id = "camera_right"
        left_info.header.stamp = timestamp
        right_info.header.stamp = timestamp
        
        # Publish all messages
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)

    def publish_equalized_images(self, left_eq, right_eq):
        """Publish histogram-equalized images to ROS topics"""
        # Convert grayscale to BGR for publishing
        if len(left_eq.shape) == 2:
            left_eq_bgr = cv.cvtColor(left_eq, cv.COLOR_GRAY2BGR)
        else:
            left_eq_bgr = left_eq
            
        if len(right_eq.shape) == 2:
            right_eq_bgr = cv.cvtColor(right_eq, cv.COLOR_GRAY2BGR)
        else:
            right_eq_bgr = right_eq

        timestamp = rospy.Time.now()
        
        left_eq_msg = self.numpy_to_ros_image(left_eq_bgr, encoding="bgr8")
        right_eq_msg = self.numpy_to_ros_image(right_eq_bgr, encoding="bgr8")
        
        left_eq_msg.header.frame_id = "camera_left"
        right_eq_msg.header.frame_id = "camera_right"
        left_eq_msg.header.stamp = timestamp
        right_eq_msg.header.stamp = timestamp
        
        self.left_eq_pub.publish(left_eq_msg)
        self.right_eq_pub.publish(right_eq_msg)

    def publish_depth_map(self, depth_map):
        """Publish RAFT-Stereo depth map to ROS topic"""
        depth_msg = Image()
        depth_msg.header = std_msgs.msg.Header()
        depth_msg.header.stamp = rospy.Time.now()
        depth_msg.header.frame_id = "camera_left"
        depth_msg.height = depth_map.shape[0]
        depth_msg.width = depth_map.shape[1]
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = depth_map.shape[1] * 4
        depth_msg.data = depth_map.astype(np.float32).tobytes()
        
        self.depth_pub.publish(depth_msg)

    def publish_pose_data(self, transform):
        """Publish ground truth pose data to ROS topic"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        
        # Position
        pose_msg.pose.position.x = transform.location.x
        pose_msg.pose.position.y = transform.location.y
        pose_msg.pose.position.z = transform.location.z
        
        # Convert Euler angles to quaternion
        rpy = np.array([transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw])
        rot = R.from_euler('xyz', rpy)
        quat = rot.as_quat()
        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)

    def imu_publisher(self):
        """Run IMU data publishing in separate thread at 200Hz"""
        rospy.loginfo("IMU publisher thread started")
        rate = rospy.Rate(200)
        
        while not rospy.is_shutdown():
            try:
                imu_data = self.get_imu_data()
                
                if imu_data is None or len(imu_data) != 6:
                    rospy.logwarn("Invalid IMU data received")
                    continue
                
                # Create IMU message
                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.header.frame_id = "base_link"
                
                # Linear acceleration (add lunar gravity)
                imu_msg.linear_acceleration.x = imu_data[0]
                imu_msg.linear_acceleration.y = imu_data[1]
                imu_msg.linear_acceleration.z = imu_data[2] + 1.62
                
                # Angular velocity
                imu_msg.angular_velocity.x = imu_data[3]
                imu_msg.angular_velocity.y = imu_data[4]
                imu_msg.angular_velocity.z = imu_data[5]
                
                # Covariance matrices
                imu_msg.orientation_covariance = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
                imu_msg.angular_velocity_covariance = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
                imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
                
                self.imu_pub.publish(imu_msg)
                rospy.loginfo_throttle(1, "IMU data published")
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"IMU publisher error: {e}")

    def run_step(self, input_data):
        """Main processing loop - stereo depth estimation and point cloud generation"""
        
        # Initialize arm positions on first frame
        if self.frame == 0:
            self.set_front_arm_angle(radians(80))
            self.set_back_arm_angle(radians(60))
            
            # Get camera-to-robot transformation
            camera_transform = self.get_camera_position(carla.SensorPosition.FrontLeft)
            self.transl_c2r = np.array([
                camera_transform.location.x, 
                camera_transform.location.y, 
                camera_transform.location.z
            ])
        
        # Get current robot pose
        robot_transform = self.get_transform()
        self.publish_pose_data(robot_transform)
        
        # Calculate transformation matrices
        rpy = np.array([
            robot_transform.rotation.roll, 
            robot_transform.rotation.pitch, 
            robot_transform.rotation.yaw
        ])
        rot = R.from_euler('xyz', rpy)
        rot_r2w = rot.as_matrix()
        transl_r2w = np.array([
            robot_transform.location.x, 
            robot_transform.location.y, 
            robot_transform.location.z
        ])
        
        transform_r2w = np.eye(4)
        transform_r2w[:3, :3] = rot_r2w
        transform_r2w[:3, 3] = transl_r2w

        # Get sensor data
        left_sensor_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        right_sensor_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        semantic_data = input_data['Semantic'].get(carla.SensorPosition.FrontLeft, None)

        if (left_sensor_data is not None and 
            right_sensor_data is not None and 
            self.get_mission_time() > 3):
            
            # Display raw images
            cv.imshow("Raw Left Image", left_sensor_data)
            
            # Apply histogram equalization
            left_enhanced = self.clhae.apply(left_sensor_data)
            right_enhanced = self.clhae.apply(right_sensor_data)
            cv.imshow("Enhanced Left Image", left_enhanced)
            
            # Publish enhanced images
            self.publish_equalized_images(left_enhanced, right_enhanced)
            
            # Convert to RGB and apply filtering
            left_rgb = cv.cvtColor(left_enhanced, cv.COLOR_GRAY2RGB)
            right_rgb = cv.cvtColor(right_enhanced, cv.COLOR_GRAY2RGB)
            
            # Apply bilateral and guided filtering
            left_filtered = cv.bilateralFilter(left_rgb, 15, 150, 150)
            right_filtered = cv.bilateralFilter(right_rgb, 15, 150, 150)
            left_filtered = cv.ximgproc.guidedFilter(left_filtered, left_filtered, radius=10, eps=1e-2)
            right_filtered = cv.ximgproc.guidedFilter(right_filtered, right_filtered, radius=10, eps=1e-2)
            
            # Ensure correct data type
            left_filtered = np.array(left_filtered.astype(np.uint8))
            right_filtered = np.array(right_filtered.astype(np.uint8))
            
            # Publish stereo images to ROS
            self.publish_stereo_images(left_filtered, right_filtered)
            
            # Compute disparity using RAFT-Stereo
            disparity = self.compute_disparity(
                self.raft_model, left_filtered, right_filtered, self.args.valid_iters
            )
            
            # Convert disparity to depth
            focal_length = 458
            baseline = 0.162
            disparity = np.where(disparity > 0, disparity, 1e-6)
            
            # Mask out earth regions from semantic data
            earth_mask_color = np.array([180, 130, 70])
            earth_mask = cv.inRange(semantic_data, earth_mask_color, earth_mask_color)
            semantic_data[earth_mask != 0] = (0, 0, 0)
            
            # Create object mask and apply to disparity
            object_mask = cv.cvtColor(semantic_data, cv.COLOR_BGR2GRAY) != 0
            disparity = np.where(object_mask, disparity, 0)
            
            # Crop disparity to region of interest
            disparity = disparity[:400, 150:490]
            
            # Visualize disparity
            disparity_viz = cv.applyColorMap(np.uint8(disparity), cv.COLORMAP_JET)
            disparity_viz[0:255, :] = (0, 0, 0)
            
            # Calculate depth map
            depth_map = (focal_length * baseline) / disparity
            depth_map[depth_map >= 1.25] = 0
            depth_map[depth_map <= 0.2] = 0
            
            # Apply median filtering to reduce noise
            depth_map = cv.medianBlur(depth_map, 5)
            
            # Apply mask to depth map
            mask = cv.cvtColor(semantic_data, cv.COLOR_BGR2GRAY) != 0
            mask = mask[:400, 150:490]
            masked_depth = np.where(mask, depth_map, 0)
            
            # Visualize depth map
            depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            depth_vis_colormap = cv.applyColorMap(-depth_vis, cv.COLORMAP_JET)
            cv.imshow("RAFT Depth Map", depth_vis_colormap)
            
            # Publish depth map
            self.publish_depth_map(depth_map)
            
            # Generate point cloud from depth map
            self._generate_point_cloud(masked_depth, focal_length, transform_r2w)
            
            self.rate.sleep()
            cv.waitKey(1)
            self.frame += 1

        # Return control command
        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        
        if self.frame >= 50000:
            self.mission_complete()
            
        return control

    def _generate_point_cloud(self, depth_map, focal_length, transform_r2w):
        """Generate and publish point cloud from depth map"""
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()
        
        valid_points = depth > 0
        Y = -((u[valid_points] - self.cx) * depth[valid_points] / focal_length)
        Z = -((v[valid_points] - self.cy) * depth[valid_points] / focal_length)
        X = depth[valid_points]
        
        # Camera coordinates
        points_cam = np.vstack((X, Y, Z)).T
        self.pcd.points = o3d.utility.Vector3dVector(points_cam)
        
        # Apply point cloud filtering
        downsampled_pcd, _ = self.pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=0.8
        )
        downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size=0.025)
        points_filtered = np.asarray(downsampled_pcd.points)
        
        # Transform to world coordinates
        points_robot = points_filtered + self.transl_c2r
        points_robot_hom = np.hstack([points_robot, np.ones((points_filtered.shape[0], 1))])
        points_world = transform_r2w @ points_robot_hom.T
        points_world = points_world[:3, :].T
        
        # Update point cloud and publish
        self.pcd.points = o3d.utility.Vector3dVector(points_world)
        cloud_msg = point_cloud2.create_cloud_xyz32(self.header, points_world)
        cloud_msg.header.stamp = self.timestamp
        self.pcd_publisher.publish(cloud_msg)

    def eul_to_rot(self, pitch, roll, yaw):
        """Convert Euler angles to rotation matrix"""
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return np.dot(R_z, np.dot(R_y, R_x))

    def finalize(self):
        """Clean up resources and update geometric map"""
        cv.destroyAllWindows()
        
        # Update geometric map with random data (placeholder)
        geometric_map = self.get_geometric_map()
        for _ in range(100):
            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())
            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)

    def on_press(self, key):
        """Handle keyboard press events for manual control"""
        if key == keyboard.Key.up:
            self.current_v += 0.3
            self.current_v = np.clip(self.current_v, 0, 0.2)
        elif key == keyboard.Key.down:
            self.current_v -= 0.3
            self.current_v = np.clip(self.current_v, -0.2, 0)
        elif key == keyboard.Key.left:
            self.current_w = 0.6
        elif key == keyboard.Key.right:
            self.current_w = -0.6
        elif key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()

    def on_release(self, key):
        """Handle keyboard release events"""
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.current_v = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.current_w = 0