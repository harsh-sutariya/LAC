#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Autonomous Data Collection Agent for Lunar Navigation World Model

This agent autonomously collects diverse trajectories for training a latent world model
as described in plan.md. It generates:
- Random walk trajectories
- Directed movement trajectories  
- Various maneuvers (turning, slope navigation, obstacle avoidance)
- RGB images, dual-timestep IMU data, 6DOF poses, and actions at each timestep

Camera synchronization: CARLA cameras run at 10Hz while simulation runs at 20Hz.
This means camera data is only available every other simulation step.
We collect data only when valid camera images are available.

IMU enhancement: Captures high-frequency IMU data from both simulation steps,
concatenating [previous_step_IMU + current_step_IMU] for richer motion dynamics.

Data is saved in a format suitable for world model training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import collections
import datetime
import json
import numpy as np
import pickle
import random
import math
import time

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

# Try to import scipy for image resizing, fallback to basic methods if not available
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - image resizing will use basic interpolation")


def get_entry_point():
    return 'DataCollectionAgent'


class TrajectoryGenerator:
    """Generates different types of autonomous trajectories for data collection"""
    
    def __init__(self, max_linear_speed=0.4, max_angular_speed=0.5):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.current_trajectory_type = None
        self.trajectory_step = 0
        self.trajectory_length = 0
        self.target_position = None
        self.start_position = None
        
        # Callback to save trajectory when starting new one
        self.save_trajectory_callback = None
        
        # Trajectory types as mentioned in plan.md
        self.trajectory_types = [
            'random_walk',
            'directed_move', 
            'turning_maneuvers',
            'exploration',
            'spiral_pattern'
        ]
    
    def set_save_callback(self, callback):
        """Set callback function to save trajectory data before starting new trajectory"""
        self.save_trajectory_callback = callback
        
    def start_new_trajectory(self, current_position):
        """Start a new trajectory of random type - saves existing data first"""
        # Save any existing trajectory data before starting fresh
        if self.save_trajectory_callback and self.current_trajectory_type is not None:
            print(f"🔄 Finishing trajectory: {self.current_trajectory_type} (completed {self.trajectory_step} steps)")
            self.save_trajectory_callback()
        
        # Start completely fresh trajectory
        self.current_trajectory_type = random.choice(self.trajectory_types)
        self.trajectory_step = 0
        self.start_position = current_position
        
        # Set trajectory length (100-1000 steps as mentioned in plan)
        self.trajectory_length = random.randint(100, 1000)
        
        if self.current_trajectory_type == 'directed_move':
            # Generate random target within map boundaries (stay within safe limits)
            current_distance_from_center = math.sqrt(current_position[0]**2 + current_position[1]**2)
            
            # Adjust target range based on current position
            if current_distance_from_center > 12.0:  # If we're getting far from center
                max_target_distance = 8.0  # Use shorter, more conservative targets
            else:
                max_target_distance = 15.0  # Normal range when safely in center
            
            max_attempts = 10
            for attempt in range(max_attempts):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(3, 15)  # Conservative range
                target_x = current_position[0] + distance * math.cos(angle)
                target_y = current_position[1] + distance * math.sin(angle)
                
                # Check if target is within safe boundaries
                target_distance_from_center = math.sqrt(target_x**2 + target_y**2)
                if target_distance_from_center <= max_target_distance:
                    self.target_position = (target_x, target_y)
                    break
            else:
                # If no safe target found, aim toward center
                center_angle = math.atan2(-current_position[1], -current_position[0])
                safe_distance = min(8.0, max_target_distance - 2.0)  # Conservative distance toward center
                self.target_position = (
                    current_position[0] + safe_distance * math.cos(center_angle),
                    current_position[1] + safe_distance * math.sin(center_angle)
                )
        
        print(f"🚀 Starting NEW trajectory: {self.current_trajectory_type} for {self.trajectory_length} steps")
        
    def get_next_action(self, current_position, current_orientation=0):
        """Generate next action based on current trajectory type"""
        if self.trajectory_step >= self.trajectory_length:
            self.start_new_trajectory(current_position)
            
        linear_speed = 0.0
        angular_speed = 0.0
        
        if self.current_trajectory_type == 'random_walk':
            # Random movements with occasional direction changes
            if self.trajectory_step % 20 == 0:  # Change direction every 20 steps
                linear_speed = random.uniform(0.05, self.max_linear_speed)
                angular_speed = random.uniform(-self.max_angular_speed, self.max_angular_speed)
            else:
                # Keep moving in roughly same direction with small variations
                linear_speed = random.uniform(0.05, self.max_linear_speed * 0.8)
                angular_speed = random.uniform(-self.max_angular_speed * 0.3, self.max_angular_speed * 0.3)
                
        elif self.current_trajectory_type == 'directed_move':
            # Move towards target position
            if self.target_position:
                dx = self.target_position[0] - current_position[0]
                dy = self.target_position[1] - current_position[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > 2.0:  # Not at target yet
                    target_angle = math.atan2(dy, dx)
                    angle_diff = target_angle - current_orientation
                    
                    # Normalize angle difference
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    linear_speed = min(self.max_linear_speed * 0.8, distance * 0.1)
                    angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, angle_diff * 2.0))
                else:
                    # Reached target, start new one
                    self.start_new_trajectory(current_position)
                    
        elif self.current_trajectory_type == 'turning_maneuvers':
            # Turning in place and curved movements
            if self.trajectory_step < self.trajectory_length // 3:
                # Turn in place
                linear_speed = 0.0
                angular_speed = random.choice([-self.max_angular_speed, self.max_angular_speed])
            elif self.trajectory_step < 2 * self.trajectory_length // 3:
                # Curved movement
                linear_speed = random.uniform(0.1, self.max_linear_speed * 0.6)
                angular_speed = random.uniform(-self.max_angular_speed * 0.7, self.max_angular_speed * 0.7)
            else:
                # Straight movement
                linear_speed = random.uniform(0.1, self.max_linear_speed)
                angular_speed = random.uniform(-0.1, 0.1)
                
        elif self.current_trajectory_type == 'exploration':
            # Exploratory behavior with stops and direction changes
            if self.trajectory_step % 50 == 0:  # Stop occasionally
                linear_speed = 0.0
                angular_speed = random.uniform(-self.max_angular_speed, self.max_angular_speed)
            else:
                linear_speed = random.uniform(0.0, self.max_linear_speed)
                angular_speed = random.uniform(-self.max_angular_speed * 0.5, self.max_angular_speed * 0.5)
                
        elif self.current_trajectory_type == 'spiral_pattern':
            # Spiral movement pattern
            t = self.trajectory_step / 100.0
            linear_speed = self.max_linear_speed * 0.5
            angular_speed = self.max_angular_speed * 0.3 * math.sin(t)
        
        self.trajectory_step += 1
        return linear_speed, angular_speed


class DataLogger:
    """Handles saving trajectory data for world model training"""
    
    def __init__(self, save_dir="data_collection", min_trajectory_length=99):
        # Use mission-specific directory if environment variable is set
        if 'MISSION_DATA_DIR' in os.environ:
            self.save_dir = os.environ['MISSION_DATA_DIR']
            print(f"🎯 Using mission-specific directory: {self.save_dir}")
        else:
            self.save_dir = save_dir
            print(f"⚠️  No mission directory specified, using default: {save_dir}")
        
        self.trajectory_count = 0
        self.total_steps_saved = 0  # Track actual total steps across all saved trajectories
        self.min_trajectory_length = min_trajectory_length  # Minimum steps to save a trajectory
        
        # Lists to store current trajectory data
        self.current_images = []
        self.current_imu_data = []
        self.current_poses = []  # Changed from positions to poses (6DOF)
        self.current_actions = []
        self.current_timestamps = []
        # self.current_vehicle_status = []
        
        # Expected data shapes for validation
        self.expected_shapes = {
            'image': (480, 640),  # Grayscale camera
            'imu': 12,  # Dual-timestep IMU: [prev 6D + curr 6D]
            'pose': 6,  # 6DOF: [x, y, z, roll, pitch, yaw]
            'action': 2  # [linear_velocity, angular_velocity]
        }
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Data will be saved to: {self.save_dir}")
        print(f"📏 Expected data shapes: image={self.expected_shapes['image']}, "
              f"imu={self.expected_shapes['imu']}D, pose={self.expected_shapes['pose']}D, "
              f"action={self.expected_shapes['action']}D")
        print(f"🎯 Minimum trajectory length: {self.min_trajectory_length} steps (shorter trajectories will be discarded)")
    
    def _validate_and_correct_imu(self, imu_data):
        """Validate and correct IMU data to ensure it's exactly 12D"""
        # Convert to list if needed
        if hasattr(imu_data, 'tolist'):
            imu_list = imu_data.tolist()
        else:
            imu_list = list(imu_data)
        
        target_size = self.expected_shapes['imu']
        current_size = len(imu_list)
        
        if current_size == target_size:
            # Perfect size
            return imu_list
        elif current_size < target_size:
            # Too short - pad by duplicating last element
            padding_needed = target_size - current_size
            if current_size > 0:
                last_element = imu_list[-1]
                padded_data = imu_list + [last_element] * padding_needed
            else:
                # Empty data - fill with zeros
                padded_data = [0.0] * target_size
            
            print(f"⚠️  IMU data corrected: {current_size}D → {target_size}D (padded)")
            return padded_data
        else:
            # Too long - truncate to target size
            truncated_data = imu_list[:target_size]
            print(f"⚠️  IMU data corrected: {current_size}D → {target_size}D (truncated)")
            return truncated_data
    
    def _validate_image(self, image_data):
        """Validate camera image data"""
        if image_data is None:
            # Create placeholder with correct shape
            placeholder = np.zeros(self.expected_shapes['image'], dtype=np.uint8)
            print("⚠️  Camera data None - using placeholder")
            return placeholder
        
        expected_shape = self.expected_shapes['image']
        if image_data.shape != expected_shape:
            print(f"⚠️  Image shape mismatch: expected {expected_shape}, got {image_data.shape}")
            # Try to resize/reshape if possible
            if len(image_data.shape) == 2 and SCIPY_AVAILABLE:
                # Already grayscale, try to resize using scipy
                resized = ndimage.zoom(image_data, 
                                     (expected_shape[0]/image_data.shape[0], 
                                      expected_shape[1]/image_data.shape[1]), 
                                     order=1)
                print(f"🔧 Image resized to {resized.shape}")
                return resized.astype(np.uint8)
            elif len(image_data.shape) == 2:
                # Basic resizing using numpy (less accurate but functional)
                resized = np.resize(image_data, expected_shape)
                print(f"🔧 Image resized (basic) to {resized.shape}")
                return resized.astype(np.uint8)
            else:
                # Create placeholder if can't fix
                placeholder = np.zeros(expected_shape, dtype=np.uint8)
                print("🔧 Using placeholder image due to shape mismatch")
                return placeholder
        
        return image_data
    
    def _validate_pose(self, pose):
        """Validate and extract 6DOF pose data"""
        try:
            pose_data = [
                pose.location.x, pose.location.y, pose.location.z,          # Position
                pose.rotation.roll, pose.rotation.pitch, pose.rotation.yaw  # Orientation
            ]
            assert len(pose_data) == self.expected_shapes['pose'], f"Pose should be {self.expected_shapes['pose']}D"
            
            # Check for NaN or infinite values
            for i, val in enumerate(pose_data):
                if not np.isfinite(val):
                    print(f"⚠️  Invalid pose value at index {i}: {val}, replacing with 0.0")
                    pose_data[i] = 0.0
            
            return pose_data
        except Exception as e:
            print(f"⚠️  Pose extraction error: {e}, using zeros")
            return [0.0] * self.expected_shapes['pose']
    
    def _validate_action(self, action):
        """Validate action data"""
        try:
            assert isinstance(action, dict), f"Action should be dict, got {type(action)}"
            assert 'linear_velocity' in action, "Action missing 'linear_velocity'"
            assert 'angular_velocity' in action, "Action missing 'angular_velocity'"
            
            linear_vel = float(action['linear_velocity'])
            angular_vel = float(action['angular_velocity'])
            
            # Check for NaN or infinite values
            if not np.isfinite(linear_vel):
                print(f"⚠️  Invalid linear velocity: {linear_vel}, replacing with 0.0")
                linear_vel = 0.0
            if not np.isfinite(angular_vel):
                print(f"⚠️  Invalid angular velocity: {angular_vel}, replacing with 0.0")
                angular_vel = 0.0
            
            action_data = [linear_vel, angular_vel]
            assert len(action_data) == self.expected_shapes['action'], f"Action should be {self.expected_shapes['action']}D"
            
            return action_data
        except Exception as e:
            print(f"⚠️  Action validation error: {e}, using zeros")
            return [0.0] * self.expected_shapes['action']
        
    def log_step(self, image_data, imu_data, pose, action, vehicle_data=None):
        """Log a single timestep of data with comprehensive validation"""
        # Store timestamp
        self.current_timestamps.append(time.time())
        
        # Validate and store image data
        validated_image = self._validate_image(image_data)
        self.current_images.append(validated_image)
        
        # Validate and correct IMU data
        corrected_imu = self._validate_and_correct_imu(imu_data)
        self.current_imu_data.append(corrected_imu)
        
        # Validate and store 6DOF pose
        validated_pose = self._validate_pose(pose)
        self.current_poses.append(validated_pose)
        
        # Validate and store action
        validated_action = self._validate_action(action)
        self.current_actions.append(validated_action)
        
        # Debug assertions (will be disabled in production but useful for development)
        if len(self.current_images) <= 5:  # Only check first few steps to avoid spam
            try:
                assert self.current_images[-1].shape == self.expected_shapes['image'], \
                    f"Image shape assertion failed: {self.current_images[-1].shape}"
                assert len(self.current_imu_data[-1]) == self.expected_shapes['imu'], \
                    f"IMU shape assertion failed: {len(self.current_imu_data[-1])}"
                assert len(self.current_poses[-1]) == self.expected_shapes['pose'], \
                    f"Pose shape assertion failed: {len(self.current_poses[-1])}"
                assert len(self.current_actions[-1]) == self.expected_shapes['action'], \
                    f"Action shape assertion failed: {len(self.current_actions[-1])}"
                
                print(f"✅ Step {len(self.current_images)}: All data shapes validated - "
                      f"img{self.current_images[-1].shape}, imu{len(self.current_imu_data[-1])}D, "
                      f"pose{len(self.current_poses[-1])}D, action{len(self.current_actions[-1])}D")
            except AssertionError as e:
                print(f"❌ Data validation failed: {e}")
                raise

    def save_trajectory(self, force_save=False):
        """Save current trajectory as a single numpy file and start a new one"""
        if len(self.current_images) >= self.min_trajectory_length or (force_save and len(self.current_images) > 0):
            current_steps = len(self.current_images)
            trajectory_filename = f"trajectory_{self.trajectory_count}.npz"
            trajectory_path = os.path.join(self.save_dir, trajectory_filename)
            
            # Convert lists to numpy arrays with final shape validation
            images = np.array(self.current_images)  # Shape: (n_steps, height, width)
            imu_data = np.array(self.current_imu_data)  # Shape: (n_steps, 12)
            poses = np.array(self.current_poses)  # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw]
            actions = np.array(self.current_actions)  # Shape: (n_steps, 2)
            timestamps = np.array(self.current_timestamps)  # Shape: (n_steps,)
            
            # Final shape validation before saving
            expected_steps = current_steps
            assert images.shape == (expected_steps, self.expected_shapes['image'][0], self.expected_shapes['image'][1]), \
                f"Images final shape error: {images.shape}"
            assert imu_data.shape == (expected_steps, self.expected_shapes['imu']), \
                f"IMU final shape error: {imu_data.shape}"
            assert poses.shape == (expected_steps, self.expected_shapes['pose']), \
                f"Poses final shape error: {poses.shape}"
            assert actions.shape == (expected_steps, self.expected_shapes['action']), \
                f"Actions final shape error: {actions.shape}"
            assert timestamps.shape == (expected_steps,), \
                f"Timestamps final shape error: {timestamps.shape}"
            
            # Save all data in a single compressed numpy file
            np.savez_compressed(
                trajectory_path,
                images=images,
                imu_data=imu_data,
                poses=poses,  # Changed from positions to poses
                actions=actions,
                timestamps=timestamps,
                # vehicle_status=vehicle_status,
                trajectory_id=self.trajectory_count,
                n_steps=current_steps
            )
            
            if force_save and current_steps < self.min_trajectory_length:
                print(f"🚨 EMERGENCY SAVE: trajectory {self.trajectory_count} with {current_steps} steps to {trajectory_filename}")
                print(f"⚠️  Below minimum length ({self.min_trajectory_length}) but force-saved due to emergency")
            else:
                print(f"✅ Saved trajectory {self.trajectory_count} with {current_steps} steps to {trajectory_filename}")
            
            print(f"📏 Final shapes: img{images.shape}, imu{imu_data.shape}, pose{poses.shape}, action{actions.shape}")
            
            # Update total steps counter with actual steps saved
            self.total_steps_saved += current_steps
            self.trajectory_count += 1
        elif len(self.current_images) > 0 and not force_save:
            print(f"🗑️  DISCARDING trajectory with {len(self.current_images)} steps (minimum: {self.min_trajectory_length})")
        else:
            print("⚠️  No data to save in current trajectory")
            
        # Clear current trajectory data regardless of whether we saved or not
        self._clear_current_data()
            
    def save_and_start_fresh(self):
        """Save current trajectory data (if any) and start completely fresh"""
        if len(self.current_images) >= self.min_trajectory_length:
            self.save_trajectory()
            print(f"📁 Saved complete trajectory before starting fresh")
        elif len(self.current_images) > 0:
            print(f"🗑️  DISCARDING partial trajectory with only {len(self.current_images)} steps (minimum: {self.min_trajectory_length})")
            self._clear_current_data()
        else:
            print(f"🔄 Starting fresh trajectory (no previous data to save)")
            self._clear_current_data()
    
    def _clear_current_data(self):
        """Clear all current trajectory data to start fresh"""
        self.current_images = []
        self.current_imu_data = []
        self.current_poses = []  # Changed from positions to poses
        self.current_actions = []
        self.current_timestamps = []
        # self.current_vehicle_status = []

    def get_statistics(self):
        """Get collection statistics"""
        current_steps = len(self.current_images)
        # Calculate actual total steps: saved trajectories + current trajectory
        actual_total_steps = self.total_steps_saved + current_steps
        
        return {
            'trajectories_saved': self.trajectory_count,
            'current_trajectory_steps': current_steps,
            'total_steps_collected': actual_total_steps,
            'total_steps_saved': self.total_steps_saved  # For debugging
        }


class DataCollectionAgent(AutonomousAgent):
    """
    Autonomous agent for collecting diverse training data for world model
    """
    
    def setup(self, path_to_conf_file):
        """Initialize the agent"""
        self.trajectory_generator = TrajectoryGenerator()
        
        # Data collection parameters
        self.target_collection_hours = 5.0  # Target collection time in hours
        self.save_frequency = 100  # Save trajectory every 100 steps
        self.initial_delay = 10.0  # Reduced to 10 seconds since camera sync is the real issue
        self.min_trajectory_length = 99  # Minimum steps to save a trajectory (discard shorter ones)
        
        # World and vehicle access for teleportation
        import carla
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            self.world = client.get_world()
            self._ego_vehicle = None
            # Store references to mission system components for vehicle respawn updates
            self._mission_spawner = None
            self._mission_manager = None
            self._mission_behaviors = None
            self._agent_wrapper = None
            # Store lander reference to preserve it during vehicle operations
            self._lander = None
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to CARLA client for teleportation: {e}")
            self.world = None
            self._ego_vehicle = None
            self._mission_spawner = None
            self._mission_manager = None
            self._mission_behaviors = None
            self._agent_wrapper = None
            self._lander = None
        
        # Initialize data logger with minimum trajectory length
        self.data_logger = DataLogger(min_trajectory_length=self.min_trajectory_length)
        
        # Connect trajectory generator to data logger for clean trajectory boundaries
        self.trajectory_generator.set_save_callback(self.data_logger.save_and_start_fresh)
        
        # Mission state
        self._mission_start_time = time.time()
        
        self._recording_started = False
        self._recording_start_time = None
        self._start_position = None
        self._arms_raised = False
        
        # Step counters
        self._simulation_step_count = 0  # Total simulation steps
        self._step_count = 0  # Logged data steps (only when camera data available)
        
        # Action and IMU tracking
        self._current_action = None
        self._previous_imu_data = None
        
        # Collision and stuck detection
        self._collision_detected = False
        self._collision_threshold = 15.0  # Acceleration threshold for collision (m/s²)
        self._stuck_detection_steps = 100  # Number of steps to check for stuck condition
        self._position_history = []  # Track recent positions for stuck detection
        self._movement_threshold = 0.05  # Minimum movement in meters over stuck_detection_steps
        self._stuck_counter = 0  # Count steps where rover is stuck
        
        # Map boundary constraints (from constants.py)
        self._map_size = 27.0  # meters (full map size)
        self._bounds_distance = 19.5  # meters (safety boundary from center to prevent mesh exit)
        self._boundary_warning_distance = 17.0  # meters (warn when approaching boundary)
        
        # Battery management
        self._battery_recharge_threshold = 100  # Wh (recharge when power drops below this)
        self._battery_min_power = 5.0  # Wh (vehicle stops moving at ~3 Wh)
        self._last_battery_status = None  # Track battery status for logging
        
        # Camera dimensions (for sensor configuration)
        self._width = 640
        self._height = 480
        
        print("🤖 Data Collection Agent initialized")
        print(f"🎯 Target collection time: {self.target_collection_hours} hours")
        print(f"⏱️  CARLA cameras run at 10Hz (every 2 sim steps). Data collected only when camera available.")
        print(f"📁 Each trajectory file will contain data from only ONE trajectory type")
        print(f"🗑️  Trajectories shorter than {self.min_trajectory_length} steps will be discarded")
        print(f"🚨 Collision detection: IMU threshold={self._collision_threshold} m/s², stuck detection={self._stuck_detection_steps} steps")
        print(f"🗺️  Map boundaries: ±{self._map_size/2:.1f}m, safety boundary: {self._bounds_distance}m radius")
        print(f"🔋 Battery management: auto-recharge below {self._battery_recharge_threshold} Wh (min: {self._battery_min_power} Wh)")

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define sensors for data collection
        Focus on front cameras as primary sensors as mentioned in plan.md
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 
                'light_intensity': 1.0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 
                'light_intensity': 1.0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 
                'light_intensity': 1.0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 
                'light_intensity': 0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 
                'light_intensity': 0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 
                'light_intensity': 0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 
                'light_intensity': 0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 
                'light_intensity': 0, 
                'width': self._width, 
                'height': self._height, 
                'use_semantic': False
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of autonomous data collection with camera-synchronized data logging"""
        
        # Raise arms and activate sensors immediately on first call
        if not self._arms_raised:
            print("🔧 Raising rover arms to clear camera view...")
            self.set_front_arm_angle(math.radians(80))  # Raise front arms
            self.set_back_arm_angle(math.radians(60))   # Raise back arms
            
            # Set light intensity (cameras should already be active from sensors() config)
            print("🔧 Setting camera lights...")
            self.set_light_state(carla.SensorPosition.Front, 1.0)
            self.set_light_state(carla.SensorPosition.FrontLeft, 1.0)
            self.set_light_state(carla.SensorPosition.FrontRight, 1.0)
            
            self._arms_raised = True
            print("✅ Arms raised and sensors configured")
        
        # Battery management - check and recharge if needed
        self._manage_battery()
        
        # Check if initial delay has passed
        elapsed_time = time.time() - self._mission_start_time
        if elapsed_time < self.initial_delay:
            # Still in waiting period - don't move and don't record
            remaining_time = self.initial_delay - elapsed_time
            if int(remaining_time) != int(remaining_time + 0.05):  # Print every second
                print(f"⏳ Camera warmup: {remaining_time:.1f}s remaining...")
            
            # Return stationary control command
            return carla.VehicleVelocityControl(0.0, 0.0)
        
        # Start recording if not already started
        if not self._recording_started:
            self._recording_started = True
            self._recording_start_time = time.time()
            print("🚀 Starting data collection! Looking for camera data...")
        
        # Increment simulation step counter
        self._simulation_step_count += 1
        
        # Get current vehicle state
        current_transform = self._vehicle_status.transform
        current_position = current_transform.location
        current_orientation = math.radians(current_transform.rotation.yaw)
        
        # Set start position on first recording step
        if self._start_position is None:
            self._start_position = current_position
            self.trajectory_generator.start_new_trajectory((current_position.x, current_position.y))
            print(f"📍 Starting data collection at position: ({current_position.x:.2f}, {current_position.y:.2f})")
        
        # Always generate an action (will be used for multiple steps if needed)
        if self._current_action is None:
            # Generate next action using trajectory generator
            linear_speed, angular_speed = self.trajectory_generator.get_next_action(
                (current_position.x, current_position.y), 
                current_orientation
            )
            
            # Store velocity values for logging
            self._current_action = {
                'linear_velocity': linear_speed,
                'angular_velocity': angular_speed,
                'generated_at_sim_step': self._simulation_step_count
            }
            
            if self._step_count < 3:
                print(f"🎮 Action generated at sim step {self._simulation_step_count}: "
                      f"linear={linear_speed:.3f}, angular={angular_speed:.3f}")
        
        # Create control command using current action
        control = carla.VehicleVelocityControl(
            self._current_action['linear_velocity'], 
            self._current_action['angular_velocity']
        )
        
        # ALWAYS capture IMU data (every simulation step for high-frequency dynamics)
        current_imu_data = self.get_imu_data()
        
        # Collision and stuck detection (runs every simulation step)
        if self._recording_started:  # Only check after recording has started
            # Detect collision using IMU data
            collision_detected = self._detect_collision(current_imu_data)
            if collision_detected:
                self._collision_detected = True
            
            # Detect if rover is stuck using position tracking
            stuck_detected = self._detect_stuck(current_position)
            
            # If both collision and stuck are detected, handle emergency exit
            if self._collision_detected or stuck_detected:
                self._handle_collision_and_stuck()
                return carla.VehicleVelocityControl(0.0, 0.0)  # Stop rover immediately
        
        # Map boundary checking (runs every simulation step)
        if self._recording_started:  # Only check after recording has started
            out_of_bounds, approaching_boundary, boundary_correction = self._check_map_boundaries(current_position)
            
            # If rover has exceeded boundary, handle emergency exit
            if out_of_bounds:
                self._handle_boundary_violation(current_position)
                return carla.VehicleVelocityControl(0.0, 0.0)  # Stop rover immediately
        
        # Check if camera data is available (CARLA 10Hz camera sync)
        front_camera_data = input_data['Grayscale'].get(carla.SensorPosition.FrontLeft, None)
        
        # Try fallback cameras if primary is None
        if front_camera_data is None:
            for fallback_camera in [carla.SensorPosition.Front, carla.SensorPosition.FrontRight]:
                front_camera_data = input_data['Grayscale'].get(fallback_camera, None)
                if front_camera_data is not None:
                    if self._step_count < 3:  # Only print first few times
                        print(f"📷 Using fallback camera: {fallback_camera.name}")
                    break
        
        # Only log data when camera data is actually available
        if front_camera_data is not None:
            # We have valid camera data - log this step
            
            # Prepare dual-timestep IMU data (previous step + current step)
            if self._previous_imu_data is not None:
                # Concatenate previous step IMU + current step IMU for richer dynamics
                if hasattr(self._previous_imu_data, 'tolist'):
                    prev_imu = self._previous_imu_data.tolist()
                else:
                    prev_imu = list(self._previous_imu_data)
                
                if hasattr(current_imu_data, 'tolist'):
                    curr_imu = current_imu_data.tolist()
                else:
                    curr_imu = list(current_imu_data)
                
                # Concatenate: [prev_acc_x, prev_acc_y, prev_acc_z, prev_gyro_x, prev_gyro_y, prev_gyro_z,
                #               curr_acc_x, curr_acc_y, curr_acc_z, curr_gyro_x, curr_gyro_y, curr_gyro_z]
                dual_imu_data = prev_imu + curr_imu
                
                if self._step_count < 3:  # Debug for first few steps
                    print(f"📊 IMU dual-step: prev={len(prev_imu)}D, curr={len(curr_imu)}D, total={len(dual_imu_data)}D")
            else:
                # First step: duplicate current IMU for consistency
                if hasattr(current_imu_data, 'tolist'):
                    curr_imu = current_imu_data.tolist()
                else:
                    curr_imu = list(current_imu_data)
                dual_imu_data = curr_imu + curr_imu  # [current, current]
                
                if self._step_count < 3:
                    print(f"📊 IMU dual-step (first): duplicated current IMU, total={len(dual_imu_data)}D")
            
            # Log camera status for first few steps
            if self._step_count < 5:
                print(f"✅ Step {self._step_count}: Camera OK, shape={front_camera_data.shape}, "
                      f"sim_step={self._simulation_step_count}")
            
            # Log the data with dual-timestep IMU
            self.data_logger.log_step(
                image_data=front_camera_data,
                imu_data=dual_imu_data,
                pose=current_transform,  # Pass full transform for 6DOF pose
                action=self._current_action,
            )
            
            self._step_count += 1
            
            # Generate new action after logging (ready for next time we have camera data)
            linear_speed, angular_speed = self.trajectory_generator.get_next_action(
                (current_position.x, current_position.y), 
                current_orientation
            )
            
            self._current_action = {
                'linear_velocity': linear_speed,
                'angular_velocity': angular_speed,
                'generated_at_sim_step': self._simulation_step_count
            }
            
            # Periodic saves and status updates
            if self._step_count % self.save_frequency == 0:
                self.data_logger.save_trajectory()
                stats = self.data_logger.get_statistics()
                # Calculate elapsed recording time (excluding initial delay)
                recording_elapsed_hours = (time.time() - self._recording_start_time) / 3600.0
                
                print(f"Step {self._step_count}: Collected {stats['trajectories_saved']} trajectories, "
                      f"{stats['total_steps_collected']} total steps, "
                      f"Recording time: {recording_elapsed_hours:.2f}h, "
                      f"Sim steps: {self._simulation_step_count}")
                
                # Check if we've collected enough data (based on recording time, not total time)
                if recording_elapsed_hours >= self.target_collection_hours:
                    print(f"Target collection time reached ({self.target_collection_hours}h). Completing mission.")
                    self.mission_complete()
        
        else:
            # No camera data available - just print debug info for first few steps
            if self._simulation_step_count % 100 == 1:  # Print every 100 simulation steps when no camera
                print(f"⏳ Sim step {self._simulation_step_count}: No camera data, continuing movement...")
        
        # ALWAYS store current IMU for next step (both logged and non-logged steps)
        self._previous_imu_data = current_imu_data
        
        return control

    def finalize(self):
        """Cleanup and final save"""
        print("Finalizing data collection...")
        self.data_logger.save_trajectory()  # Save any remaining data
        
        stats = self.data_logger.get_statistics()
        total_elapsed_hours = (time.time() - self._mission_start_time) / 3600.0
        
        # Calculate actual recording time (excluding initial delay)
        if hasattr(self, '_recording_start_time'):
            recording_hours = (time.time() - self._recording_start_time) / 3600.0
        else:
            recording_hours = 0.0  # In case finalize is called before recording started
        
        print(f"Data collection complete!")
        print(f"Total mission time: {total_elapsed_hours:.2f} hours (including {self.initial_delay}s startup delay)")
        print(f"Actual recording time: {recording_hours:.2f} hours")
        print(f"Trajectories collected: {stats['trajectories_saved']}")
        print(f"Total logged steps: {stats['total_steps_collected']} (when camera data available)")
        print(f"Total simulation steps: {self._simulation_step_count}")
        print(f"Camera sync ratio: {self._step_count / self._simulation_step_count * 100:.1f}% of sim steps had camera data")
        print(f"Data saved to: {self.data_logger.save_dir}")
        
        # Create mission-specific summary file
        mission_summary = {
            'mission_data_dir': self.data_logger.save_dir,
            'total_mission_duration_hours': total_elapsed_hours,
            'recording_duration_hours': recording_hours,
            'initial_delay_seconds': self.initial_delay,
            'total_trajectories': stats['trajectories_saved'],
            'total_logged_steps': stats['total_steps_collected'],
            'total_simulation_steps': self._simulation_step_count,
            'camera_sync_ratio': self._step_count / max(self._simulation_step_count, 1),
            'synchronization_note': 'Data logged only when CARLA camera data available (10Hz camera, 20Hz sim)',
            'imu_enhancement': 'Dual-timestep IMU: [previous_step + current_step] for richer dynamics',
            'image_resolution': (self._width, self._height),
            'trajectory_types': self.trajectory_generator.trajectory_types,
            'completion_time': datetime.datetime.now().isoformat()
        }
        
        # Save mission-specific summary
        mission_summary_path = os.path.join(self.data_logger.save_dir, "mission_summary.json")
        with open(mission_summary_path, 'w') as f:
            json.dump(mission_summary, f, indent=2)
        
        print(f"Mission summary saved to: {mission_summary_path}")
        
        # Create/update global summary in main data_collection directory
        self._update_global_summary()

    def _detect_collision(self, imu_data):
        """
        Detect collision using IMU acceleration data.
        Returns True if collision is detected.
        """
        if imu_data is None or len(imu_data) < 6:
            return False
        
        try:
            # Extract current step acceleration (last 3 values for current step)
            if len(imu_data) >= 12:
                # Dual-timestep IMU: current acceleration is at indices 6, 7, 8
                acc_x, acc_y, acc_z = imu_data[6], imu_data[7], imu_data[8]
            else:
                # Legacy single-step IMU: acceleration is at indices 0, 1, 2
                acc_x, acc_y, acc_z = imu_data[0], imu_data[1], imu_data[2]
            
            # Calculate total acceleration magnitude
            acceleration_magnitude = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # Check if acceleration exceeds collision threshold
            if acceleration_magnitude > self._collision_threshold:
                print(f"🚨 COLLISION DETECTED! Acceleration magnitude: {acceleration_magnitude:.2f} m/s² (threshold: {self._collision_threshold})")
                return True
                
        except Exception as e:
            print(f"⚠️  Error in collision detection: {e}")
            
        return False
    
    def _detect_stuck(self, current_position):
        """
        Detect if rover is stuck by tracking position changes over recent steps.
        Returns True if rover hasn't moved much over the last stuck_detection_steps.
        """
        # Add current position to history
        self._position_history.append((current_position.x, current_position.y, current_position.z))
        
        # Keep only the last stuck_detection_steps positions
        if len(self._position_history) > self._stuck_detection_steps:
            self._position_history.pop(0)
        
        # Need at least stuck_detection_steps positions to check
        if len(self._position_history) < self._stuck_detection_steps:
            return False
        
        # Calculate total movement over the tracked period
        start_pos = np.array(self._position_history[0])
        end_pos = np.array(self._position_history[-1])
        total_movement = np.linalg.norm(end_pos - start_pos)
        
        # Check if movement is below threshold
        is_stuck = total_movement < self._movement_threshold
        
        if is_stuck:
            self._stuck_counter += 1
            if self._stuck_counter % 5 == 1:  # Print every 5 steps to avoid spam
                print(f"🔒 Rover appears stuck: movement={total_movement:.3f}m over {len(self._position_history)} steps "
                      f"(threshold: {self._movement_threshold}m)")
        else:
            self._stuck_counter = 0  # Reset counter if rover moves
            
        return is_stuck
    
    def _handle_collision_and_stuck(self):
        """
        Handle the case where rover has collided and is stuck.
        Save current trajectory and raise exception to trigger mission restart.
        """
        print("💥 COLLISION + STUCK DETECTED!")
        print(f"🔒 Rover has been stuck for {self._stuck_counter} steps after collision")
        print(f"📁 Force-saving current trajectory before mission restart (ignoring minimum length)...")
        
        # Force save current trajectory data regardless of length
        self.data_logger.save_trajectory(force_save=True)
        
        # Update global summary before restarting
        self._update_global_summary()
        
        # Get current statistics
        stats = self.data_logger.get_statistics()
        collision_location = self._position_history[-1] if self._position_history else (0, 0, 0)
        
        print(f"🔄 RESTARTING MISSION (raising exception to trigger restart)")
        print(f"📊 Current Progress:")
        print(f"   Trajectories collected so far: {stats['trajectories_saved']}")
        print(f"   Total logged steps: {stats['total_steps_collected']}")
        print(f"   Collision location: ({collision_location[0]:.2f}, {collision_location[1]:.2f}, {collision_location[2]:.2f})")
        
        # Raise exception to trigger mission restart instead of ending mission
        from leaderboard.agents.agent_wrapper import AgentRuntimeError
        raise AgentRuntimeError("Collision and stuck detected - restarting mission")
    
    def _manage_battery(self):
        """
        Monitor battery level and automatically recharge when below threshold.
        Uses the same approach as testing_agent.py.
        """
        try:
            current_power = self.get_current_power()
            
            # Check if we need to recharge
            if current_power <= self._battery_recharge_threshold:
                # Get vehicle reference if not already available
                if self._ego_vehicle is None and self.world is not None:
                    try:
                        vehicles = self.world.get_actors().filter('vehicle.ipex.ipex')
                        if len(vehicles) > 0:
                            self._ego_vehicle = vehicles[0]
                    except Exception as e:
                        print(f"⚠️  Error finding rover vehicle for battery recharge: {e}")
                
                # Perform recharge
                if self._ego_vehicle is not None:
                    try:
                        self._ego_vehicle.recharge_battery()
                        
                        # Log recharge event (avoid spam by checking if status changed)
                        if self._last_battery_status != "recharging":
                            print(f"🔋 BATTERY RECHARGE: {current_power:.1f} Wh → Full charge (threshold: {self._battery_recharge_threshold} Wh)")
                            self._last_battery_status = "recharging"
                            
                    except Exception as e:
                        print(f"⚠️  Error recharging battery: {e}")
                else:
                    if self._last_battery_status != "no_vehicle":
                        print(f"⚠️  Battery low ({current_power:.1f} Wh) but no vehicle reference for recharging")
                        self._last_battery_status = "no_vehicle"
            else:
                # Battery is above threshold - reset status for future logging
                if self._last_battery_status in ["recharging", "no_vehicle"]:
                    self._last_battery_status = "normal"
            
            # Warn if battery is critically low (shouldn't happen with auto-recharge)
            if current_power <= self._battery_min_power:
                print(f"🚨 CRITICAL BATTERY WARNING: {current_power:.1f} Wh (rover may stop moving!)")
                
        except Exception as e:
            print(f"⚠️  Error in battery management: {e}")
    
    def _check_map_boundaries(self, current_position):
        """
        Check if rover is approaching or exceeding map boundaries.
        Returns (out_of_bounds, approaching_boundary, corrected_action)
        """
        # Calculate distance from center (0, 0)
        distance_from_center = math.sqrt(current_position.x**2 + current_position.y**2)
        
        # Check if rover is outside safety boundary
        out_of_bounds = distance_from_center > self._bounds_distance
        
        # Check if rover is approaching boundary
        approaching_boundary = distance_from_center > self._boundary_warning_distance
        
        # Generate corrected action if needed (stop and steer toward center)
        corrected_action = None
        if approaching_boundary:
            # Calculate angle toward center
            angle_to_center = math.atan2(-current_position.y, -current_position.x)
            
            # STOP and steer toward center when approaching boundary
            linear_speed = 0.0  # Stop forward movement
            angular_speed = angle_to_center * 1.0  # More aggressive steering toward center
            
            corrected_action = {
                'linear_velocity': linear_speed,
                'angular_velocity': max(-0.5, min(0.5, angular_speed)),  # Increased turn rate for faster correction
                'generated_at_sim_step': self._simulation_step_count,
                'boundary_correction': True
            }
        
        return out_of_bounds, approaching_boundary, corrected_action
    
    def _handle_boundary_violation(self, current_position):
        """
        Handle the case where rover has exceeded map boundaries.
        Save current trajectory and raise exception to trigger mission restart.
        """
        distance_from_center = math.sqrt(current_position.x**2 + current_position.y**2)
        
        print("🗺️  MAP BOUNDARY VIOLATION!")
        print(f"🚨 Rover exceeded safety boundary: {distance_from_center:.2f}m (limit: {self._bounds_distance}m)")
        print(f"📍 Current position: ({current_position.x:.2f}, {current_position.y:.2f})")
        print(f"📁 Force-saving current trajectory before mission restart (ignoring minimum length)...")
        
        # Force save current trajectory data regardless of length
        self.data_logger.save_trajectory(force_save=True)
        
        # Update global summary before restarting
        self._update_global_summary()
        
        # Get current statistics
        stats = self.data_logger.get_statistics()
        
        print(f"🔄 RESTARTING MISSION (raising exception to trigger restart)")
        print(f"📊 Current Progress:")
        print(f"   Trajectories collected so far: {stats['trajectories_saved']}")
        print(f"   Total logged steps: {stats['total_steps_collected']}")
        print(f"   Boundary violation location: ({current_position.x:.2f}, {current_position.y:.2f})")
        
        # Raise exception to trigger mission restart instead of ending mission
        from leaderboard.agents.agent_wrapper import AgentRuntimeError
        raise AgentRuntimeError("Boundary violation detected - restarting mission")
    
    def _update_global_summary(self):
        """Create or update global summary across all missions"""
        try:
            # Get the main data_collection directory (parent of current mission directory)
            main_data_dir = os.path.dirname(self.data_logger.save_dir)
            if not main_data_dir.endswith('data_collection'):
                # Fallback to default if not in expected structure
                main_data_dir = "data_collection"
            
            # Find all mission directories
            mission_dirs = []
            if os.path.exists(main_data_dir):
                for item in os.listdir(main_data_dir):
                    item_path = os.path.join(main_data_dir, item)
                    if os.path.isdir(item_path) and item.startswith('mission_'):
                        mission_dirs.append(item_path)
            
            # Collect statistics from all missions
            total_trajectories = 0
            total_logged_steps = 0
            total_simulation_steps = 0
            mission_details = []
            
            for mission_dir in sorted(mission_dirs):
                mission_name = os.path.basename(mission_dir)
                
                # Count trajectories in this mission
                mission_trajectories = len([f for f in os.listdir(mission_dir) if f.endswith('.npz')])
                total_trajectories += mission_trajectories
                
                # Read mission summary if available
                mission_summary_path = os.path.join(mission_dir, "mission_summary.json")
                if os.path.exists(mission_summary_path):
                    try:
                        with open(mission_summary_path, 'r') as f:
                            mission_data = json.load(f)
                        
                        total_logged_steps += mission_data.get('total_logged_steps', 0)
                        total_simulation_steps += mission_data.get('total_simulation_steps', 0)
                        
                        mission_details.append({
                            'mission_name': mission_name,
                            'trajectories': mission_trajectories,
                            'logged_steps': mission_data.get('total_logged_steps', 0),
                            'simulation_steps': mission_data.get('total_simulation_steps', 0),
                            'duration_hours': mission_data.get('recording_duration_hours', 0),
                            'completion_time': mission_data.get('completion_time', 'unknown')
                        })
                    except Exception as e:
                        print(f"⚠️  Error reading mission summary {mission_summary_path}: {e}")
                        mission_details.append({
                            'mission_name': mission_name,
                            'trajectories': mission_trajectories,
                            'logged_steps': 0,
                            'simulation_steps': 0,
                            'duration_hours': 0,
                            'completion_time': 'unknown'
                        })
                else:
                    mission_details.append({
                        'mission_name': mission_name,
                        'trajectories': mission_trajectories,
                        'logged_steps': 0,
                        'simulation_steps': 0,
                        'duration_hours': 0,
                        'completion_time': 'unknown'
                    })
            
            # Create global summary
            global_summary = {
                'total_missions': len(mission_dirs),
                'total_trajectories': total_trajectories,
                'total_logged_steps': total_logged_steps,
                'total_simulation_steps': total_simulation_steps,
                'camera_sync_ratio': total_logged_steps / max(total_simulation_steps, 1),
                'total_recording_hours': sum(m.get('duration_hours', 0) for m in mission_details),
                'mission_details': mission_details,
                'last_updated': datetime.datetime.now().isoformat(),
                'data_structure': 'Each mission has its own directory (mission_1, mission_2, etc.)',
                'synchronization_note': 'Data logged only when CARLA camera data available (10Hz camera, 20Hz sim)',
                'imu_enhancement': 'Dual-timestep IMU: [previous_step + current_step] for richer dynamics'
            }
            
            # Save global summary
            global_summary_path = os.path.join(main_data_dir, "collection_summary.json")
            with open(global_summary_path, 'w') as f:
                json.dump(global_summary, f, indent=2)
            
            print(f"Global summary updated: {global_summary_path}")
            print(f"📊 Total across all missions: {total_trajectories} trajectories, {total_logged_steps} logged steps")
            
        except Exception as e:
            print(f"⚠️  Error updating global summary: {e}")
    
 