# Lunar Navigation Data Collection System

This system implements autonomous data collection for training a latent world model for lunar rover navigation, as described in `plan.md`. It collects diverse trajectories with RGB images, dual-timestep IMU data, 6DOF poses, and actions for training the world model.

## Overview

The data collection system consists of three main components:

1. **`agents/data_collection_agent.py`** - Autonomous agent that collects training data with comprehensive validation
2. **Leaderboard execution** - Run with `./RunLeaderboard.sh --agent=agents.data_collection_agent`
3. **`load_trajectory_example.py`** - Analysis and visualization of collected data

### 🚨 Safety Features: Collision Detection & Emergency Exit

The system includes robust safety mechanisms to handle collision scenarios:

- **IMU-based collision detection**: Monitors acceleration magnitude to detect impacts with rocks/obstacles
- **Stuck detection**: Tracks position changes over 70 consecutive steps to identify when rover is immobilized
- **Map boundary enforcement**: Prevents rover from exiting the 27m×27m map area with safety boundaries
- **Emergency restart protocol**: When collision+stuck OR boundary violation occurs, the system:
  - **Force-saves current trajectory data immediately** (ignoring minimum length requirement)
  - **Teleports rover to a safe location** (2-8m from map center with random orientation)
  - **Resets all detection states** and starts new trajectory
  - **Continues mission** for maximum data collection instead of exiting
- **Configurable thresholds**: 
  - Collision threshold: 15.0 m/s² acceleration magnitude
  - Stuck threshold: <0.05m movement over 70 steps
  - Map boundary: 19.5m radius safety limit, 17.0m warning threshold
- **Real-time monitoring**: All safety checks run every simulation step for immediate response
- **Boundary avoidance**: When approaching boundaries, rover STOPS forward movement and steers toward map center

## Quick Start

### 1. Run Data Collection

```bash
# Run the data collection with the leaderboard system (5-hour collection)
./RunLeaderboard.sh --agent=agents.data_collection_agent

# The agent will automatically:
# - Collect diverse trajectory types (random_walk, directed_move, turning_maneuvers, etc.)
# - Run for 5 hours of recording time (configurable)
# - Save only trajectories with ≥99 logged steps (shorter ones are discarded)
# - Ensure each trajectory file contains only ONE trajectory type
# - Validate and correct all data shapes automatically
# - Handle collisions and boundary violations with automatic recovery
```

### Collection Duration Configuration

The current collection target is set to **5 hours** in `agents/data_collection_agent.py`:

```python
self.target_collection_hours = 5.0  # Target collection time in hours
```

To modify the collection duration:
- **Shorter collection**: Change to `2.0` for 2 hours
- **Longer collection**: Change to `10.0` for 10 hours (as recommended in plan.md)
- **Full dataset**: Change to `20.0` for 20 hours for comprehensive training data

### 2. Analyze Collected Data

```bash
# Analyze and visualize the collected data
python load_trajectory_example.py

# This will show:
# - Trajectory statistics and quality metrics
# - Dual-timestep IMU analysis
# - 6DOF pose visualization
# - World model training data examples
```

## Data Collection Strategy

Based on the plan in `plan.md`, the system implements diverse trajectory generation with strict quality control:

### Trajectory Types

1. **Random Walk** - Random movements with occasional direction changes
2. **Directed Move** - Goal-directed movement toward random waypoints
3. **Turning Maneuvers** - In-place turning and curved movements
4. **Exploration** - Exploratory behavior with stops and direction changes
5. **Spiral Pattern** - Systematic spiral movement patterns

### Data Quality Control

- **Minimum Length**: Only trajectories with ≥99 logged data points are saved (bypassed in emergencies)
- **Pure Trajectory Types**: Each file contains data from only ONE trajectory type (no mixing)
- **Automatic Validation**: All data shapes are validated and corrected automatically
- **Camera Synchronization**: Data logged only when CARLA camera data is available (10Hz)
- **Emergency Preservation**: Collision/boundary violations force-save any collected data regardless of length
- **Automatic Recovery**: Rover teleports to safe locations and continues mission instead of terminating

### Data Collected

For each timestep where camera data is available, the system records:

- **RGB Image** (640x480) from front camera (grayscale, validated)
- **Dual-Timestep IMU Data** (12D) - [prev_acc_xyz, prev_gyro_xyz, curr_acc_xyz, curr_gyro_xyz]
- **6DOF Pose** (x, y, z, roll, pitch, yaw) - complete pose information
- **Action** (linear and angular velocities commanded, validated)
- **Timestamp** for temporal analysis

## File Structure

After collection, your data directory will look like:

```
data_collection/
├── trajectory_0.npz           # Pure trajectory type (e.g., turning_maneuvers)
├── trajectory_1.npz           # Pure trajectory type (e.g., random_walk)
├── trajectory_2.npz           # Pure trajectory type (e.g., directed_move)
├── trajectory_3.npz           # Pure trajectory type (e.g., exploration)
└── ...
└── collection_summary.json    # Overall collection statistics
```

Each `.npz` file contains validated data for a single trajectory type:
- `images`: (n_steps, height, width) array - validated grayscale images
- `imu_data`: (n_steps, 12) array - DUAL-TIMESTEP IMU [prev 6D + curr 6D]
- `poses`: (n_steps, 6) array - [x, y, z, roll, pitch, yaw] 6DOF poses
- `actions`: (n_steps, 2) array - [linear_vel, angular_vel] validated actions
- `timestamps`: (n_steps,) array - Unix timestamps
- `trajectory_id`: Scalar trajectory number
- `n_steps`: Number of logged steps (≥99 guaranteed)

## Configuration

### Data Collection Agent Parameters

You can modify these parameters in `agents/data_collection_agent.py`:

```python
# Collection duration
self.target_collection_hours = 5.0

# Save frequency (steps between trajectory saves)
self.save_frequency = 100

# Minimum trajectory length (shorter trajectories discarded)
self.min_trajectory_length = 99

# Image resolution
self._width = 640
self._height = 480

# Trajectory generation parameters
max_linear_speed = 0.4    # m/s
max_angular_speed = 0.5   # rad/s
```

### Data Validation Settings

The system automatically validates and corrects:

```python
# Expected data shapes (enforced automatically)
expected_shapes = {
    'image': (480, 640),  # Grayscale camera
    'imu': 12,           # Dual-timestep IMU
    'pose': 6,           # 6DOF pose
    'action': 2          # Linear + angular velocity
}
```

### Camera Synchronization

- **CARLA cameras**: Run at 10Hz (every ~2 simulation steps)
- **Simulation**: Runs at 20Hz 
- **Data logging**: Only when camera data is available (~50% of simulation steps)
- **IMU collection**: Every simulation step (dual-timestep enhancement)

## Data Format

### Enhanced Trajectory NPZ Format

Each trajectory is saved as a single compressed NumPy file (`.npz`) with validated data:

```python
# Load trajectory data
data = np.load('trajectory_0.npz')

# Available arrays (all validated):
images = data['images']           # Shape: (n_steps, 480, 640) - validated grayscale
imu_data = data['imu_data']       # Shape: (n_steps, 12) - DUAL-TIMESTEP IMU
poses = data['poses']             # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw]
actions = data['actions']         # Shape: (n_steps, 2) - [linear_vel, angular_vel]
timestamps = data['timestamps']   # Shape: (n_steps,) - Unix timestamps
trajectory_id = data['trajectory_id']    # Scalar: trajectory number
n_steps = data['n_steps']         # Scalar: number of steps (≥99 guaranteed)

# Dual-timestep IMU breakdown:
prev_accelerometer = imu_data[:, 0:3]   # Previous step acceleration
prev_gyroscope = imu_data[:, 3:6]       # Previous step angular velocity  
curr_accelerometer = imu_data[:, 6:9]   # Current step acceleration
curr_gyroscope = imu_data[:, 9:12]      # Current step angular velocity

# 6DOF pose breakdown:
positions = poses[:, :3]         # [x, y, z] positions
orientations = poses[:, 3:]      # [roll, pitch, yaw] orientations
```

### Advantages of Enhanced Format

- **Quality Guaranteed**: Only substantial trajectories (≥99 steps) are saved
- **Pure Data**: Each file contains only one trajectory type
- **Validated Shapes**: All data automatically validated and corrected
- **Rich Dynamics**: Dual-timestep IMU captures high-frequency motion
- **Complete Pose**: 6DOF pose enables full state reconstruction
- **Fast Loading**: NumPy arrays load directly into memory
- **Compressed Storage**: Reduced disk space usage

## Requirements

### System Requirements

- CARLA Simulator with lunar environment
- Python 3.7+
- Sufficient disk space (estimate ~10 GB per hour)

### Python Dependencies

```bash
pip install numpy matplotlib scipy
# scipy is used for image resizing (fallback to numpy if unavailable)
```

## Usage Examples

### Basic Data Collection

```bash
# 1. Start CARLA with lunar environment
# 2. Run collection (will run for 5 hours)
./RunLeaderboard.sh --agent=agents.data_collection_agent

# System will automatically:
# - Collect diverse trajectory types for 5 hours
# - Validate all data shapes
# - Save only quality trajectories (≥99 steps)
# - Discard partial/incomplete trajectories
# - Handle collisions and boundary violations with automatic recovery
```

### Data Analysis

```bash
# Analyze collected data with enhanced visualization
python load_trajectory_example.py

# Shows:
# - Dual-timestep IMU analysis
# - 6DOF pose trajectories with orientation arrows
# - Data quality validation results
# - World model training examples
```

## Expected Data Volume and Quality

Based on the plan.md targets with quality control:

- **Target**: 5 hours for current collection (configurable up to 10-20 hours for full dataset)
- **Quality**: Only trajectories with ≥99 logged steps
- **Purity**: Each file contains only one trajectory type
- **Frame Rate**: ~5-10 Hz (when camera data available)
- **Total Frames**: ~90k-180k for 5 hours at 5-10 Hz
- **Estimated Size**: ~5 GB for 5 hours

## Integration with World Model Training

The collected data is optimized for world model training:

1. **Enhanced IMU**: Dual-timestep data provides richer motion dynamics
2. **Complete Pose**: 6DOF pose enables full state modeling
3. **Validated Data**: All shapes guaranteed correct, no preprocessing needed
4. **Pure Trajectories**: Each file contains consistent behavior type
5. **Quality Control**: Only substantial trajectories (≥99 steps)

Example enhanced loading code:

```python
import numpy as np

# Load validated trajectory
data = np.load('trajectory_0.npz')

# All data is pre-validated and ready for training
images = data['images']        # Shape: (n_steps, 480, 640) - guaranteed
dual_imu = data['imu_data']    # Shape: (n_steps, 12) - dual-timestep
poses = data['poses']          # Shape: (n_steps, 6) - complete 6DOF
actions = data['actions']      # Shape: (n_steps, 2) - validated

# Extract dual-timestep IMU components
prev_imu = dual_imu[:, :6]     # Previous step IMU
curr_imu = dual_imu[:, 6:]     # Current step IMU

# Extract pose components
positions = poses[:, :3]       # [x, y, z] 
orientations = poses[:, 3:]    # [roll, pitch, yaw]

# Create training sequences (example: 10-step sequences)
sequence_length = 10
n_sequences = len(images) - sequence_length + 1

# All sequences guaranteed to have proper shapes
image_sequences = np.array([images[i:i+sequence_length] for i in range(n_sequences)])
action_sequences = np.array([actions[i:i+sequence_length] for i in range(n_sequences)])
pose_sequences = np.array([poses[i:i+sequence_length] for i in range(n_sequences)])
# Result shapes: (n_sequences, sequence_length, data_dims)
```

## Data Validation and Error Handling

The system includes comprehensive validation:

### Automatic Corrections

1. **IMU Data**: 
   - Too short → Padded by duplicating last element
   - Too long → Truncated to 12D
   - Invalid values → Replaced with safe defaults

2. **Images**:
   - None data → Placeholder with correct shape
   - Wrong shape → Resized using scipy/numpy
   - Invalid format → Safe fallbacks

3. **Poses**:
   - NaN/Infinite → Replaced with 0.0
   - Extraction errors → Zero pose fallback

4. **Actions**:
   - Missing keys → Error handling with defaults
   - Invalid values → Safe fallbacks

### Quality Assurance

- **Shape Assertions**: Debug output for first few steps
- **Minimum Length**: Trajectories <99 steps automatically discarded
- **Real-time Monitoring**: Status messages show validation results
- **Error Recovery**: System continues even with data issues

## Troubleshooting

### Common Issues

1. **Partial Trajectories**: System automatically discards trajectories <99 steps
2. **Camera Sync**: ~50% of simulation steps have camera data (this is normal)
3. **Data Shapes**: All shapes automatically validated and corrected
4. **Performance**: Validation adds minimal overhead

### Monitoring Collection

Enhanced status messages show:

```
🗑️ DISCARDING partial trajectory with only 45 steps (minimum: 99)
✅ Saved trajectory 0 with 156 steps to trajectory_0.npz
📏 Final shapes: img(156, 480, 640), imu(156, 12), pose(156, 6), action(156, 2)
```

### Data Quality Validation

Use the enhanced analysis script:

```bash
python load_trajectory_example.py
```

This shows:
- Data quality summary
- Shape validation results
- Dual-timestep IMU analysis
- Trajectory type purity
- Distance and orientation analysis

## Next Steps

After collecting validated data, proceed to:

1. **World Model Training** - Use dual-timestep IMU and 6DOF poses for richer models
2. **Planning Implementation** - Leverage complete state information for better planning
3. **Evaluation** - Test navigation with enhanced state representation
  
  The collected data is now optimized for world model training with guaranteed quality and consistency. See `plan.md` for detailed implementation guidelines for the world model and planning components. 