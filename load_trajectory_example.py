#!/usr/bin/env python3
"""
Example script showing how to load trajectory data saved in the new format.

Each trajectory is saved as a single .npz file containing all data:
- images: (n_steps, height, width) array
- imu_data: (n_steps, 6) array  
- poses: (n_steps, 6) array - [x, y, z, roll, pitch, yaw]
- actions: (n_steps, 2) array
- timestamps: (n_steps,) array
# - vehicle_status: (n_steps, 4) array
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_trajectory(trajectory_path):
    """
    Load a single trajectory file.
    
    Returns:
        dict: Dictionary containing all trajectory data as numpy arrays
    """
    data = np.load(trajectory_path)
    
    trajectory_info = {
        'images': data['images'],           # Shape: (n_steps, height, width)
        'imu_data': data['imu_data'],       # Shape: (n_steps, 6) - [ax, ay, az, gx, gy, gz]
        'poses': data['poses'],             # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw]
        'actions': data['actions'],         # Shape: (n_steps, 2) - [linear_vel, angular_vel]
        'timestamps': data['timestamps'],   # Shape: (n_steps,)
        # 'vehicle_status': data['vehicle_status'],  # Shape: (n_steps, 4)
        'trajectory_id': data['trajectory_id'].item(),
        'n_steps': data['n_steps'].item()
    }
    
    # Handle legacy files that still have 'positions' instead of 'poses'
    if 'positions' in data and 'poses' not in data:
        # Convert 3D positions to 6D poses by adding zero orientation
        positions = data['positions']
        trajectory_info['poses'] = np.concatenate([
            positions,  # x, y, z
            np.zeros((positions.shape[0], 3))  # roll, pitch, yaw = 0
        ], axis=1)
        print("Warning: Legacy trajectory file with positions only. Orientation set to zero.")
    
    # Handle vehicle_status for backward compatibility with old data files
    if 'vehicle_status' in data:
        trajectory_info['vehicle_status'] = data['vehicle_status']
    
    return trajectory_info

def load_all_trajectories(data_dir):
    """
    Load all trajectory files from a directory.
    
    Returns:
        list: List of trajectory dictionaries
    """
    trajectories = []
    
    # Find all .npz files in the directory
    trajectory_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    trajectory_files.sort()  # Sort by filename for consistent ordering
    
    for filename in trajectory_files:
        filepath = os.path.join(data_dir, filename)
        try:
            traj = load_trajectory(filepath)
            trajectories.append(traj)
            print(f"Loaded {filename}: {traj['n_steps']} steps")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return trajectories

def create_world_model_sequences(trajectory, sequence_length=10):
    """
    Create sequences for world model training from a trajectory.
    
    Args:
        trajectory: Trajectory dictionary from load_trajectory()
        sequence_length: Length of sequences to create
        
    Returns:
        dict: Dictionary with sequences ready for training
    """
    n_steps = trajectory['n_steps']
    if n_steps < sequence_length:
        return None
    
    # Create overlapping sequences
    sequences = {
        'images': [],
        'imu_data': [],
        'poses': [],
        'actions': []
    }
    
    for i in range(n_steps - sequence_length + 1):
        sequences['images'].append(trajectory['images'][i:i+sequence_length])
        sequences['imu_data'].append(trajectory['imu_data'][i:i+sequence_length])
        sequences['poses'].append(trajectory['poses'][i:i+sequence_length])
        sequences['actions'].append(trajectory['actions'][i:i+sequence_length])
    
    # Convert to numpy arrays
    for key in sequences:
        sequences[key] = np.array(sequences[key])
    
    return sequences

def analyze_trajectory(trajectory):
    """Print analysis of a trajectory."""
    print(f"\nTrajectory Analysis:")
    print(f"Trajectory ID: {trajectory['trajectory_id']}")
    print(f"Number of steps: {trajectory['n_steps']}")
    print(f"Duration: {trajectory['timestamps'][-1] - trajectory['timestamps'][0]:.2f} seconds")
    
    print(f"\nImage data:")
    print(f"  Shape: {trajectory['images'].shape}")
    print(f"  Data type: {trajectory['images'].dtype}")
    print(f"  Value range: [{trajectory['images'].min()}, {trajectory['images'].max()}]")
    print(f"  Mean value: {trajectory['images'].mean():.2f}")
    print(f"  Non-zero pixels: {np.count_nonzero(trajectory['images'])}/{trajectory['images'].size} ({100*np.count_nonzero(trajectory['images'])/trajectory['images'].size:.1f}%)")
    
    print(f"\nPose range:")
    poses = trajectory['poses']
    print(f"  Position X: [{poses[:, 0].min():.2f}, {poses[:, 0].max():.2f}]")
    print(f"  Position Y: [{poses[:, 1].min():.2f}, {poses[:, 1].max():.2f}]")
    print(f"  Position Z: [{poses[:, 2].min():.2f}, {poses[:, 2].max():.2f}]")
    print(f"  Roll:       [{np.degrees(poses[:, 3]).min():.1f}°, {np.degrees(poses[:, 3]).max():.1f}°]")
    print(f"  Pitch:      [{np.degrees(poses[:, 4]).min():.1f}°, {np.degrees(poses[:, 4]).max():.1f}°]")
    print(f"  Yaw:        [{np.degrees(poses[:, 5]).min():.1f}°, {np.degrees(poses[:, 5]).max():.1f}°]")
    
    print(f"\nAction range:")
    actions = trajectory['actions']
    print(f"  Linear velocity: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
    print(f"  Angular velocity: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
    
    print(f"\nIMU data range:")
    imu = trajectory['imu_data']
    print(f"  Accelerometer X: [{imu[:, 0].min():.3f}, {imu[:, 0].max():.3f}]")
    print(f"  Accelerometer Y: [{imu[:, 1].min():.3f}, {imu[:, 1].max():.3f}]")
    print(f"  Accelerometer Z: [{imu[:, 2].min():.3f}, {imu[:, 2].max():.3f}]")

def visualize_trajectory(trajectory, save_path=None):
    """Create visualizations of the trajectory data."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Increased to 3x3 for more plots
    
    # Extract pose data
    poses = trajectory['poses']
    positions = poses[:, :3]  # x, y, z
    orientations = poses[:, 3:]  # roll, pitch, yaw
    
    # 1. Trajectory path (top view) with orientation arrows
    axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
    axes[0, 0].scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
    axes[0, 0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
    
    # Add orientation arrows (every 10th point to avoid clutter)
    step = max(1, len(positions) // 20)
    for i in range(0, len(positions), step):
        x, y = positions[i, 0], positions[i, 1]
        yaw = orientations[i, 2]  # yaw angle
        dx, dy = 0.5 * np.cos(yaw), 0.5 * np.sin(yaw)
        axes[0, 0].arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, 
                        fc='orange', ec='orange', alpha=0.7)
    
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Trajectory Path with Orientation')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    
    # 2. Actions over time
    actions = trajectory['actions']
    time_steps = np.arange(len(actions))
    axes[0, 1].plot(time_steps, actions[:, 0], 'b-', label='Linear velocity')
    axes[0, 1].plot(time_steps, actions[:, 1], 'r-', label='Angular velocity')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Actions Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Sample image
    if trajectory['images'].shape[0] > 0:
        sample_img = trajectory['images'][0]
        im = axes[0, 2].imshow(sample_img, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Sample Image (First Frame)')
        axes[0, 2].axis('off')
        # Add a small colorbar to show the intensity range
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. IMU acceleration
    imu = trajectory['imu_data']
    axes[1, 0].plot(time_steps, imu[:, 0], 'r-', label='Accel X')
    axes[1, 0].plot(time_steps, imu[:, 1], 'g-', label='Accel Y')
    axes[1, 0].plot(time_steps, imu[:, 2], 'b-', label='Accel Z')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Acceleration')
    axes[1, 0].set_title('IMU Accelerometer Data')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. IMU gyroscope
    axes[1, 1].plot(time_steps, imu[:, 3], 'r-', label='Gyro X')
    axes[1, 1].plot(time_steps, imu[:, 4], 'g-', label='Gyro Y')
    axes[1, 1].plot(time_steps, imu[:, 5], 'b-', label='Gyro Z')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Angular Velocity')
    axes[1, 1].set_title('IMU Gyroscope Data')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Z position over time
    axes[1, 2].plot(time_steps, positions[:, 2], 'purple')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Z Position (m)')
    axes[1, 2].set_title('Height Over Time')
    axes[1, 2].grid(True)
    
    # 7. Roll, Pitch, Yaw over time
    axes[2, 0].plot(time_steps, np.degrees(orientations[:, 0]), 'r-', label='Roll')
    axes[2, 0].plot(time_steps, np.degrees(orientations[:, 1]), 'g-', label='Pitch')
    axes[2, 0].plot(time_steps, np.degrees(orientations[:, 2]), 'b-', label='Yaw')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Angle (degrees)')
    axes[2, 0].set_title('Orientation Over Time')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # 8. 3D trajectory plot
    ax_3d = fig.add_subplot(3, 3, 8, projection='3d')
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7)
    ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100)
    ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100)
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory')
    
    # 9. Distance traveled over time
    distances = np.cumsum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    distances = np.concatenate([[0], distances])  # Add zero at start
    axes[2, 2].plot(time_steps, distances, 'orange')
    axes[2, 2].set_xlabel('Time Step')
    axes[2, 2].set_ylabel('Cumulative Distance (m)')
    axes[2, 2].set_title('Distance Traveled')
    axes[2, 2].grid(True)
    
    # Hide the unused subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def main():
    """Example usage of the trajectory loading functions."""
    data_dir = "data_collection"
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        print("Make sure you have run the data collection first.")
        return
    
    print("Loading trajectories...")
    trajectories = load_all_trajectories(data_dir)
    
    if not trajectories:
        print("No trajectory files found!")
        return
    
    print(f"\nLoaded {len(trajectories)} trajectories")
    
    # Analyze first trajectory
    first_traj = trajectories[0]
    analyze_trajectory(first_traj)
    
    # Create training sequences
    print(f"\nCreating training sequences...")
    sequences = create_world_model_sequences(first_traj, sequence_length=10)
    if sequences:
        print(f"Created {len(sequences['images'])} sequences of length 10")
        print(f"Sequence shapes:")
        for key, value in sequences.items():
            print(f"  {key}: {value.shape}")
    
    # Visualize trajectory
    print(f"\nCreating visualization...")
    try:
        fig = visualize_trajectory(first_traj, f"trajectory_{first_traj['trajectory_id']}_analysis.png")
        plt.show()
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Example: prepare data for world model training
    print(f"\nExample data for world model training:")
    print("State at time t:")
    print(f"  Image shape: {first_traj['images'][0].shape}")
    print(f"  IMU data: {first_traj['imu_data'][0]}")
    print(f"  Pose: {first_traj['poses'][0]}")  # [x, y, z, roll, pitch, yaw]
    print("Action at time t:")
    print(f"  Action: {first_traj['actions'][0]}")
    print("Next state at time t+1:")
    print(f"  Next pose: {first_traj['poses'][1]}")  # [x, y, z, roll, pitch, yaw]
    print(f"  Next IMU: {first_traj['imu_data'][1]}")

if __name__ == "__main__":
    main() 