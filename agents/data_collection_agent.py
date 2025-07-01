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
- RGB images, IMU data, positions, and actions at each timestep

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
        
        # Trajectory types as mentioned in plan.md
        self.trajectory_types = [
            'random_walk',
            'directed_move', 
            'turning_maneuvers',
            'exploration',
            'spiral_pattern'
        ]
        
    def start_new_trajectory(self, current_position):
        """Start a new trajectory of random type"""
        self.current_trajectory_type = random.choice(self.trajectory_types)
        self.trajectory_step = 0
        self.start_position = current_position
        
        # Set trajectory length (100-1000 steps as mentioned in plan)
        self.trajectory_length = random.randint(100, 1000)
        
        if self.current_trajectory_type == 'directed_move':
            # Generate random target within reasonable range
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(10, 50)  # 10-50 meters
            self.target_position = (
                current_position[0] + distance * math.cos(angle),
                current_position[1] + distance * math.sin(angle)
            )
        
        print(f"Starting new trajectory: {self.current_trajectory_type} for {self.trajectory_length} steps")
        
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
    
    def __init__(self, save_dir="data_collection"):
        self.save_dir = save_dir
        self.trajectory_count = 0
        self.total_steps_saved = 0  # Track actual total steps across all saved trajectories
        
        # Lists to store current trajectory data
        self.current_images = []
        self.current_imu_data = []
        self.current_poses = []  # Changed from positions to poses (6DOF)
        self.current_actions = []
        self.current_timestamps = []
        # self.current_vehicle_status = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Data will be saved to: {save_dir}")
        
    def log_step(self, image_data, imu_data, pose, action, vehicle_data=None):
        """Log a single timestep of data"""
        # Store timestamp
        self.current_timestamps.append(time.time())
        
        # Store image (handle None case)
        if image_data is not None:
            self.current_images.append(image_data)
        else:
            # Store empty array with correct shape for consistency
            self.current_images.append(np.zeros((480, 640), dtype=np.uint8))
        
        # Store IMU data
        if hasattr(imu_data, 'tolist'):
            self.current_imu_data.append(imu_data.tolist())
        else:
            self.current_imu_data.append(list(imu_data))
        
        # Store 6DOF pose (position + orientation)
        self.current_poses.append([
            pose.location.x, pose.location.y, pose.location.z,          # Position
            pose.rotation.roll, pose.rotation.pitch, pose.rotation.yaw  # Orientation
        ])
        
        # Store action
        self.current_actions.append([action['linear_velocity'], action['angular_velocity']])
        
        # # Store vehicle status
        # self.current_vehicle_status.append([
        #     vehicle_data.front_drums_speed,
        #     vehicle_data.front_arm_angle, 
        #     vehicle_data.back_arm_angle,
        #     vehicle_data.back_drums_speed
        # ])
        
    def save_trajectory(self):
        """Save current trajectory as a single numpy file and start a new one"""
        if len(self.current_images) > 0:
            current_steps = len(self.current_images)
            trajectory_filename = f"trajectory_{self.trajectory_count}.npz"
            trajectory_path = os.path.join(self.save_dir, trajectory_filename)
            
            # Convert lists to numpy arrays
            images = np.array(self.current_images)  # Shape: (n_steps, height, width)
            imu_data = np.array(self.current_imu_data)  # Shape: (n_steps, 6)
            poses = np.array(self.current_poses)  # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw]
            actions = np.array(self.current_actions)  # Shape: (n_steps, 2)
            timestamps = np.array(self.current_timestamps)  # Shape: (n_steps,)
            # vehicle_status = np.array(self.current_vehicle_status)  # Shape: (n_steps, 4)
            
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
            
            print(f"Saved trajectory {self.trajectory_count} with {current_steps} steps to {trajectory_filename}")
            
            # Update total steps counter with actual steps saved
            self.total_steps_saved += current_steps
            
            # Clear current trajectory data
            self.current_images = []
            self.current_imu_data = []
            self.current_poses = []  # Changed from positions to poses
            self.current_actions = []
            self.current_timestamps = []
            # self.current_vehicle_status = []
            
            self.trajectory_count += 1
            
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
    Autonomous agent for collecting diverse trajectory data for world model training
    """

    def setup(self, path_to_conf_file):
        """Setup the agent parameters"""
        self._width = 640  # Reduced resolution for faster collection
        self._height = 480
        
        self.trajectory_generator = TrajectoryGenerator()
        self.data_logger = DataLogger()
        
        self._step_count = 0
        self._start_position = None
        self._mission_start_time = time.time()
        self._recording_started = False
        self._arms_raised = False
        
        # Collection parameters
        self.target_collection_hours = 2.0  # Start with 2 hours, increase as needed
        self.save_frequency = 100  # Save trajectory every 100 steps
        self.initial_delay = 10.0  # Wait 10 seconds before starting data collection
        
        print("🤖 Data Collection Agent initialized")
        print(f"🎯 Target collection time: {self.target_collection_hours} hours")
        print(f"⏱️  Sequence: Raise arms → Activate sensors → Wait {self.initial_delay}s → Start collecting")

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define sensors for data collection
        Focus on front camera as primary sensor as mentioned in plan.md
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 
                'light_intensity': 0, 
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
                'camera_active': False, 
                'light_intensity': 0, 
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
        """Execute one step of autonomous data collection"""
        
        # Raise arms and activate sensors immediately on first call
        if not self._arms_raised:
            print("🔧 Raising rover arms to clear camera view...")
            self.set_front_arm_angle(math.radians(80))  # Raise front arms
            self.set_back_arm_angle(math.radians(60))   # Raise back arms
            
            # Activate the FrontLeft camera explicitly
            print("🔧 Activating FrontLeft camera...")
            self.set_camera_state(carla.SensorPosition.FrontLeft, True)
            self.set_light_state(carla.SensorPosition.FrontLeft, 1.0)
            
            self._arms_raised = True
            print("✅ Arms raised and sensors activated")
        
        # Check if initial delay has passed
        elapsed_time = time.time() - self._mission_start_time
        if elapsed_time < self.initial_delay:
            # Still in waiting period - don't move and don't record
            remaining_time = self.initial_delay - elapsed_time
            if int(remaining_time) != int(remaining_time + 0.05):  # Print every second
                print(f"⏳ Waiting {remaining_time:.1f} seconds before starting data collection...")
            
            # Return stationary control command
            return carla.VehicleVelocityControl(0.0, 0.0)
        
        # Start recording if not already started
        if not self._recording_started:
            self._recording_started = True
            self._recording_start_time = time.time()
            print("🚀 Starting data collection now!")
        
        # Get current vehicle state
        current_transform = self._vehicle_status.transform
        current_position = current_transform.location
        current_orientation = math.radians(current_transform.rotation.yaw)
        
        # Set start position on first recording step
        if self._start_position is None:
            self._start_position = current_position
            self.trajectory_generator.start_new_trajectory((current_position.x, current_position.y))
            print(f"📍 Starting data collection at position: ({current_position.x:.2f}, {current_position.y:.2f})")
            
        # Generate next action using trajectory generator
        linear_speed, angular_speed = self.trajectory_generator.get_next_action(
            (current_position.x, current_position.y), 
            current_orientation
        )
        
        # Store velocity values for logging
        action_data = {
            'linear_velocity': linear_speed,
            'angular_velocity': angular_speed
        }
        
        # Create control command
        control = carla.VehicleVelocityControl(linear_speed, angular_speed)
        
        # Log data
        front_camera_data = input_data['Grayscale'].get(carla.SensorPosition.FrontLeft, None)
        imu_data = self.get_imu_data()
        
        # Debug camera data (print only occasionally)
        if self._step_count % 50 == 0:  # Print every 50 steps
            if front_camera_data is not None:
                print(f"📷 Camera data: shape={front_camera_data.shape}, "
                      f"range=[{front_camera_data.min()}-{front_camera_data.max()}], "
                      f"mean={front_camera_data.mean():.1f}")
            else:
                print("⚠️  WARNING: Camera data is None!")
        
        self.data_logger.log_step(
            image_data=front_camera_data,
            imu_data=imu_data,
            pose=current_transform,  # Pass full transform for 6DOF pose
            action=action_data,
            # vehicle_data=self._vehicle_status
        )
        
        self._step_count += 1
        
        # Periodic saves and status updates
        if self._step_count % self.save_frequency == 0:
            self.data_logger.save_trajectory()
            stats = self.data_logger.get_statistics()
            # Calculate elapsed recording time (excluding initial delay)
            recording_elapsed_hours = (time.time() - self._recording_start_time) / 3600.0
            
            print(f"Step {self._step_count}: Collected {stats['trajectories_saved']} trajectories, "
                  f"{stats['total_steps_collected']} total steps, "
                  f"Recording time: {recording_elapsed_hours:.2f}h")
            
            # Check if we've collected enough data (based on recording time, not total time)
            if recording_elapsed_hours >= self.target_collection_hours:
                print(f"Target collection time reached ({self.target_collection_hours}h). Completing mission.")
                self.mission_complete()
        
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
        print(f"Total steps: {stats['total_steps_collected']}")
        print(f"Data saved to: {self.data_logger.save_dir}")
        
        # Create summary file
        summary = {
            'total_mission_duration_hours': total_elapsed_hours,
            'recording_duration_hours': recording_hours,
            'initial_delay_seconds': self.initial_delay,
            'total_trajectories': stats['trajectories_saved'],
            'total_steps': stats['total_steps_collected'],
            'image_resolution': (self._width, self._height),
            'trajectory_types': self.trajectory_generator.trajectory_types,
            'completion_time': datetime.datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.data_logger.save_dir, "collection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Collection summary saved to: {summary_path}") 