# Lunar Navigation Data Collection System

This system implements autonomous data collection for training a latent world model for lunar rover navigation, as described in `plan.md`. It collects diverse trajectories with RGB images, IMU data, positions, and actions for training the world model.

## Overview

The data collection system consists of three main components:

1. **`agents/data_collection_agent.py`** - Autonomous agent that collects training data
2. **`run_data_collection.py`** - Configuration and execution script
3. **`analyze_collected_data.py`** - Analysis and visualization of collected data

## Quick Start

### 1. Run Data Collection

```bash
# Basic usage (2 hours of data collection)
python run_data_collection.py

# Custom collection time and output directory
python run_data_collection.py --hours 5.0 --output-dir my_lunar_data

# Run the actual collection with the leaderboard
./RunLeaderboard.sh --agent=agents.data_collection_agent
```

### 2. Analyze Collected Data

```bash
# Analyze and visualize the collected data
python analyze_collected_data.py data_collection/

# Custom output directory for analysis
python analyze_collected_data.py data_collection/ --output-dir analysis_results/
```

## Data Collection Strategy

Based on the plan in `plan.md`, the system implements diverse trajectory generation:

### Trajectory Types

1. **Random Walk** - Random movements with occasional direction changes
2. **Directed Move** - Goal-directed movement toward random waypoints
3. **Turning Maneuvers** - In-place turning and curved movements
4. **Exploration** - Exploratory behavior with stops and direction changes
5. **Spiral Pattern** - Systematic spiral movement patterns

### Data Collected

For each timestep, the system records:

- **RGB Image** (640x480) from front camera
- **IMU Data** (accelerometer + gyroscope, 6 values)
- **Position** (x, y, z coordinates relative to start)
- **Action** (linear and angular velocities commanded)
- **Vehicle Status** (drum speeds, arm angles, etc.)
- **Timestamp** for temporal analysis

## File Structure

After collection, your data directory will look like:

```
data_collection/
├── trajectory_0.npz           # Complete trajectory 0 (all data)
├── trajectory_1.npz           # Complete trajectory 1 (all data)
├── trajectory_2.npz           # Complete trajectory 2 (all data)
└── ...
└── collection_summary.json    # Overall collection statistics
```

Each `.npz` file contains all data for that trajectory:
- `images`: (n_steps, height, width) array
- `imu_data`: (n_steps, 6) array  
- `poses`: (n_steps, 6) array - [x, y, z, roll, pitch, yaw]
- `actions`: (n_steps, 2) array
- `timestamps`: (n_steps,) array
- `vehicle_status`: (n_steps, 4) array (removed for efficiency)

## Configuration

### Data Collection Agent Parameters

You can modify these parameters in `agents/data_collection_agent.py`:

```python
# Collection duration
self.target_collection_hours = 2.0

# Save frequency (steps between trajectory saves)
self.save_frequency = 100

# Image resolution
self._width = 640
self._height = 480

# Trajectory generation parameters
max_linear_speed = 0.4    # m/s
max_angular_speed = 0.5   # rad/s
```

### Trajectory Length

- Random trajectory lengths between 100-1000 steps
- Each step corresponds to one simulation timestep
- At ~10 Hz, this gives 10-100 second trajectory segments

## Data Format

### Trajectory NPZ Format

Each trajectory is saved as a single compressed NumPy file (`.npz`) containing:

```python
# Load trajectory data
data = np.load('trajectory_0.npz')

# Available arrays:
images = data['images']           # Shape: (n_steps, 480, 640) - grayscale images
imu_data = data['imu_data']       # Shape: (n_steps, 6) - [ax, ay, az, gx, gy, gz]
poses = data['poses']             # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw] in world coordinates
actions = data['actions']         # Shape: (n_steps, 2) - [linear_vel, angular_vel]
timestamps = data['timestamps']   # Shape: (n_steps,) - Unix timestamps
# vehicle_status removed for focus on core world model data
trajectory_id = data['trajectory_id']    # Scalar: trajectory number
n_steps = data['n_steps']         # Scalar: number of steps in trajectory
```

### Advantages of NPZ Format

- **Single file per trajectory**: All data together for easy loading
- **Compressed storage**: Reduced disk space usage
- **Fast loading**: NumPy arrays load directly into memory
- **Type preservation**: Maintains data types (uint8 for images, float64 for positions)
- **Easy slicing**: Direct access to sequences for training

## Requirements

### System Requirements

- CARLA Simulator with lunar environment
- Python 3.7+
- Sufficient disk space (estimate ~10 GB per hour)

### Python Dependencies

```bash
pip install numpy matplotlib pandas
```

## Usage Examples

### Basic Data Collection

```bash
# 1. Start CARLA with lunar environment
# 2. Run collection for 2 hours
python run_data_collection.py

# 3. Execute with leaderboard
./RunLeaderboard.sh --agent=agents.data_collection_agent
```

### Extended Collection

```bash
# Collect 10 hours of data in custom directory
python run_data_collection.py --hours 10.0 --output-dir lunar_dataset_large

# Run with leaderboard
./RunLeaderboard.sh --agent=agents.data_collection_agent
```

### Data Analysis

```bash
# Analyze collected data
python analyze_collected_data.py lunar_dataset_large/

# Skip plots if matplotlib not available
python analyze_collected_data.py lunar_dataset_large/ --no-plots

# Load and visualize trajectory data (new format)
python load_trajectory_example.py
```

## Expected Data Volume

Based on the plan.md targets:

- **Initial Goal**: 2-10 hours of driving data
- **Target**: 10-20 hours for full dataset
- **Estimated Size**: ~10 GB per hour
- **Frame Rate**: ~5-10 Hz
- **Total Frames**: ~360k for 10 hours at 10 Hz

## Integration with World Model Training

The collected data is structured for easy integration with world model training:

1. **Images** can be loaded and preprocessed for encoder training
2. **IMU + 6DOF Pose** data provides complete state information
3. **Actions** provide supervision for dynamics learning  
4. **Trajectories** are pre-segmented for temporal modeling

Example loading code:

```python
import numpy as np

# Load complete trajectory (single file)
data = np.load('trajectory_0.npz')

# All data is immediately available as numpy arrays
images = data['images']        # Shape: (n_steps, 480, 640)
poses = data['poses']          # Shape: (n_steps, 6) - [x, y, z, roll, pitch, yaw]
actions = data['actions']      # Shape: (n_steps, 2) 
imu_data = data['imu_data']    # Shape: (n_steps, 6)

# Extract position and orientation separately if needed
positions = poses[:, :3]       # Shape: (n_steps, 3) - [x, y, z]
orientations = poses[:, 3:]    # Shape: (n_steps, 3) - [roll, pitch, yaw]

# Create sequences for world model training (example: 10-step sequences)
sequence_length = 10
n_sequences = len(images) - sequence_length + 1

image_sequences = np.array([images[i:i+sequence_length] for i in range(n_sequences)])
action_sequences = np.array([actions[i:i+sequence_length] for i in range(n_sequences)])
# Result: image_sequences.shape = (n_sequences, sequence_length, 480, 640)
```

## Troubleshooting

### Common Issues

1. **Disk Space**: Monitor available space during collection
2. **Performance**: Reduce image resolution or collection frequency if needed
3. **CARLA Connection**: Ensure CARLA is running before starting collection

### Monitoring Collection

The agent prints regular status updates:

```
Step 1000: Collected 10 trajectories, 1000 total steps, Elapsed: 0.25h
```

### Data Validation

Use the analysis script to verify data quality:

```bash
python analyze_collected_data.py data_collection/
```

This will show:
- Trajectory length distributions
- Velocity distributions  
- Spatial coverage
- Data completeness

## Next Steps

After collecting data, proceed to:

1. **World Model Training** - Use collected data to train encoder and dynamics model
2. **Planning Implementation** - Implement MPC planner using learned model
3. **Evaluation** - Test navigation performance on goal-reaching tasks

See `plan.md` for detailed implementation guidelines for the world model and planning components. 