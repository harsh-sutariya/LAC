# Lunar Navigation Data Collection System

This system implements autonomous data collection for training a latent world model for lunar rover navigation, as described in `plan.md`. It collects diverse trajectories with RGB images, dual-timestep IMU data, 6DOF poses, and actions for training the world model.

## Overview

The data collection system consists of two main components:

1. **`RunContinuousDataCollection.sh`** - Bash script that runs missions in a continuous loop with automatic restart on failures
2. **`agents/data_collection_agent.py`** - Autonomous agent that collects training data with comprehensive validation

The system automatically restarts missions when collisions, boundary violations, or other failures occur, ensuring maximum data collection with diverse environments.

### 🚨 Safety Features: Collision Detection & Emergency Restart

The system includes robust safety mechanisms to handle collision scenarios:

- **IMU-based collision detection**: Monitors acceleration magnitude to detect impacts with rocks/obstacles
- **Stuck detection**: Tracks position changes over 100 consecutive steps to identify when rover is immobilized
- **Map boundary enforcement**: Monitors rover position within the 27m×27m map area with safety boundaries
- **Emergency restart protocol**: When collision+stuck OR boundary violation occurs, the system:
  - **Force-saves current trajectory data immediately** (ignoring minimum length requirement)
  - **Restarts the entire mission** with a new random seed and map/preset combination
  - **Creates a new mission directory** to avoid overwriting previous data
  - **Initializes simulator** before each restart for clean state
- **Configurable thresholds**: 
  - Collision threshold: 15.0 m/s² acceleration magnitude
  - Stuck threshold: <0.05m movement over 100 steps
  - Map boundary: 19.5m radius safety limit (no steering correction - allows boundary crossing)
- **Real-time monitoring**: All safety checks run every simulation step for immediate response
- **Mission diversity**: Each restart uses randomized maps, presets, and spawn points for maximum data variety

## Quick Start

### 1. Run Continuous Data Collection

```bash
# Run the continuous data collection system
./RunContinuousDataCollection.sh

# The system will automatically:
# - Run missions continuously until target trajectory count is reached
# - Use randomized maps (Moon_Map_01, Moon_Map_02) and presets for diversity
# - Create separate directories for each mission attempt
# - Restart missions on collisions, boundary violations, or other failures
# - Initialize simulator before each restart for clean state
# - Collect diverse trajectory types (random_walk, directed_move, turning_maneuvers, etc.)
# - Save only trajectories with ≥99 logged steps (shorter ones are discarded)
# - Ensure each trajectory file contains only ONE trajectory type
# - Validate and correct all data shapes automatically
# - Handle failures with automatic mission restart
```

### Collection Target Configuration

The current collection target is set to **1000 trajectories** in `RunContinuousDataCollection.sh`:

```bash
# Continue if we have less than 1000 trajectories (adjust as needed)
if [ $trajectory_count -lt 1000 ]; then
    return 0  # Continue
else
    echo "🎯 Target trajectory count reached! Stopping continuous collection."
    return 1  # Stop
fi
```

To modify the collection target:
- **Shorter collection**: Change to `500` for 500 trajectories
- **Longer collection**: Change to `2000` for 2000 trajectories
- **Full dataset**: Change to `5000` for comprehensive training data

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

Based on the plan in `plan.md`, the system implements diverse trajectory generation with strict quality control and maximum environment variety:

### Mission Diversity

The system ensures maximum data diversity through:

1. **Randomized Maps**: Cycles through `Moon_Map_01` and `Moon_Map_02`
2. **Randomized Presets**: 
   - `Moon_Map_01`: Presets 0-10 (11 options)
   - `Moon_Map_02`: Presets 11-13 (3 options)
3. **Randomized Seeds**: Each mission uses timestamp + attempt number for unique spawn points
4. **Mission Isolation**: Each mission gets its own directory to prevent data overwrites

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
- **Mission Restart**: Failed missions restart with new environment and spawn point instead of continuing
- **Data Isolation**: Each mission's data is saved in separate directories to prevent overwrites

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
├── mission_1/                 # First mission attempt
│   ├── trajectory_0.npz       # Pure trajectory type (e.g., turning_maneuvers)
│   ├── trajectory_1.npz       # Pure trajectory type (e.g., random_walk)
│   └── ...
├── mission_2/                 # Second mission attempt (different map/preset)
│   ├── trajectory_0.npz       # Pure trajectory type (e.g., directed_move)
│   ├── trajectory_1.npz       # Pure trajectory type (e.g., exploration)
│   └── ...
├── mission_3/                 # Third mission attempt
│   └── ...
└── collection_summary.json    # Overall collection statistics across all missions
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
# Save frequency (steps between trajectory saves)
self.save_frequency = 100

# Minimum trajectory length (shorter trajectories discarded)
self.min_trajectory_length = 99

# Stuck detection threshold (steps)
self.stuck_threshold = 100

# Image resolution
self._width = 640
self._height = 480

# Trajectory generation parameters
max_linear_speed = 0.4    # m/s
max_angular_speed = 0.5   # rad/s
```

### Continuous Collection Parameters

You can modify these parameters in `RunContinuousDataCollection.sh`:

```bash
# Target trajectory count (when to stop collection)
if [ $trajectory_count -lt 1000 ]; then
    return 0  # Continue
fi

# Available maps and presets
AVAILABLE_MAPS=("Moon_Map_01" "Moon_Map_02")
MOON_MAP_01_PRESETS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
MOON_MAP_02_PRESETS=("11" "12" "13")
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
# Run continuous data collection (will run until target trajectory count reached)
./RunContinuousDataCollection.sh

# System will automatically:
# - Run missions continuously with randomized maps and presets
# - Create separate directories for each mission attempt
# - Restart missions on failures with simulator initialization
# - Collect diverse trajectory types across different environments
# - Validate all data shapes
# - Save only quality trajectories (≥99 steps)
# - Discard partial/incomplete trajectories
# - Handle failures with automatic mission restart
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

- **Target**: 1000 trajectories for current collection (configurable up to 2000-5000 for full dataset)
- **Quality**: Only trajectories with ≥99 logged steps
- **Purity**: Each file contains only one trajectory type
- **Diversity**: Multiple maps (Moon_Map_01, Moon_Map_02) with different presets
- **Frame Rate**: ~5-10 Hz (when camera data available)
- **Total Frames**: ~99k-990k for 1000 trajectories at 99-990 steps each
- **Estimated Size**: ~10-50 GB for 1000 trajectories (depending on trajectory lengths)
- **Mission Isolation**: Each mission's data stored in separate directories

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
5. **Mission Failures**: System automatically restarts with new environment (this is expected)
6. **Simulator Issues**: System reinitializes simulator before each restart

### Monitoring Collection

Enhanced status messages show:

```
🔄 Continuous Data Collection - Attempt 1
🚀 Starting leaderboard execution (attempt 1)...
🗺️  Using map: Moon_Map_01 with preset: 5
📁 Mission data directory: /path/to/data_collection/mission_1
🎲 Using random seed: 1703123456
🗑️ DISCARDING partial trajectory with only 45 steps (minimum: 99)
✅ Saved trajectory 0 with 156 steps to trajectory_0.npz
📏 Final shapes: img(156, 480, 640), imu(156, 12), pose(156, 6), action(156, 2)
⚠️  Leaderboard detected failure (exit code: 1)
🔄 This is expected for collision/boundary violations - restarting...
🚀 Initializing Lunar Simulator before restart...
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