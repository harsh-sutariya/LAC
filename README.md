# Lunar Autonomy Challenge

An advanced autonomous navigation system for lunar surface exploration using state-of-the-art computer vision and AI techniques.

## Overview

This project implements intelligent autonomous agents for lunar rover navigation using:

- **RAFT-Stereo Neural Networks** for accurate depth estimation
- **Real-time Point Cloud Processing** with Open3D
- **DBSCAN Clustering** for obstacle detection and avoidance
- **PID-Controlled Navigation** with spiral path planning
- **ROS Integration** for real-time visualization
- **CARLA Simulation** for realistic testing environments

## Project Structure

```
LunarAutonomyChallenge/
├── agents/                     # Autonomous agent implementations
│   ├── core/                   # RAFT-Stereo core modules
│   ├── dummy_agent.py          # Simple baseline agent
│   ├── human_agent.py          # Manual control agent
│   ├── opencv_agent.py         # OpenCV-based agent
│   ├── picd.py                 # RAFT-Stereo autonomous agent
│   └── raft.py                 # Advanced RAFT-Stereo agent
├── Leaderboard/                # Evaluation framework
├── LunarSimulator/             # CARLA-based lunar simulation
├── ORB_SLAM3/                  # ORB-SLAM3 submodule for SLAM functionality
├── models/                     # Pre-trained neural network weights
├── docs/                       # Documentation and fiducials
├── docker/                     # Docker configuration files
└── requirements.txt            # Python dependencies
```

## Quick Start

### 0. Clone Repository with Submodules

**⚠️ Important:** This repository contains submodules (ORB-SLAM3). Use one of these methods to clone:

**Method 1: Clone with submodules (Recommended)**
```bash
git clone --recursive https://github.com/[your-username]/LunarAutonomyChallenge.git
cd LunarAutonomyChallenge
```

**Method 2: Clone then initialize submodules**
```bash
git clone https://github.com/[your-username]/LunarAutonomyChallenge.git
cd LunarAutonomyChallenge
git submodule update --init --recursive
```

**Method 3: If you already cloned without submodules**
```bash
cd LunarAutonomyChallenge
git submodule update --init --recursive
```

### 1. Create Conda Environment

Set up the required Python environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate lunar-autonomy
```

### 2. Build ORB-SLAM3 (Optional)

If you plan to use SLAM functionality, build the ORB-SLAM3 submodule:

```bash
cd ORB_SLAM3
# Install dependencies (see Dependencies.md in ORB_SLAM3 folder)
chmod +x build.sh
./build.sh

# For ROS support (optional)
chmod +x build_ros.sh
./build_ros.sh

cd ..
```

**Note:** ORB-SLAM3 has its own dependencies (OpenCV, Eigen3, Pangolin, etc.). 
Check `ORB_SLAM3/Dependencies.md` for detailed installation instructions.

### 3. Launch Lunar Simulator

Start the CARLA-based lunar simulation environment:

```bash
bash RunLunarSimulator.sh
```

This will:
- Initialize the CARLA simulation server
- Load the lunar surface environment
- Set up sensor configurations
- Prepare the rover for autonomous navigation

### 4. Run Leaderboard Evaluation

Execute the autonomous navigation challenge:

```bash
bash RunLeaderboard.sh
```

This will:
- Launch the leaderboard evaluation framework
- Load your selected autonomous agent
- Begin the navigation mission
- Record performance metrics and results

## Available Agents

### RAFTStereoAgent (`picd.py`)
- **Primary Agent**: Advanced stereo vision with RAFT-Stereo neural networks
- **Features**: Real-time depth estimation, point cloud generation, obstacle avoidance
- **Best For**: Accurate navigation in complex lunar terrain

### RAFTStereoAgent (`raft.py`) 
- **Enhanced Version**: Comprehensive stereo processing with advanced filtering
- **Features**: Multi-threaded processing, semantic segmentation, ROS publishing
- **Best For**: Research and development with full sensor suite

### OpenCVAgent (`opencv_agent.py`)
- **Classical Approach**: Traditional computer vision techniques
- **Features**: SGBM stereo matching, basic obstacle detection
- **Best For**: Baseline comparisons and educational purposes

### DummyAgent (`dummy_agent.py`)
- **Simple Agent**: Basic movement patterns
- **Features**: Predefined navigation without vision
- **Best For**: Testing simulation framework

### HumanAgent (`human_agent.py`)
- **Manual Control**: Keyboard-based rover control
- **Features**: Real-time manual navigation
- **Best For**: Data collection and validation

## Configuration

### Model Weights

Ensure RAFT-Stereo model weights are available:
```bash
# Download pre-trained weights (if not included)
wget -O models/raftstereo-realtime.pth [MODEL_URL]
```

### Agent Selection

Modify the agent configuration in your run scripts or set environment variables:
```bash
export AGENT_MODULE="picd"  # or "raft", "opencv_agent", etc.
```

### Sensor Configuration

Agents can be configured with different sensor setups:
- **Stereo Cameras**: Front-left and front-right for depth estimation
- **Semantic Segmentation**: Object classification for terrain analysis
- **IMU**: Inertial measurement for pose estimation
- **RGB Cameras**: Visual navigation and debugging

## Performance Optimization

### GPU Acceleration
- Ensure CUDA is available for RAFT-Stereo inference
- Configure GPU memory allocation in agent settings
- Monitor GPU utilization during execution

### Point Cloud Processing
- Adjust voxel downsampling parameters for performance
- Tune DBSCAN clustering parameters for obstacle detection
- Configure statistical outlier removal thresholds

### Navigation Parameters
- Modify PID controller gains for different terrain types
- Adjust obstacle avoidance distances based on rover dimensions
- Configure spiral path planning parameters

## Troubleshooting

### Common Issues

**CUDA Extension Errors:**
```bash
# If RAFT-Stereo CUDA extensions fail to load
# The agents automatically fall back to PyTorch implementations
# Check core/corr.py for fallback handling
```

**Missing Dependencies:**
```bash
# Install additional dependencies if needed
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Simulation Connection:**
```bash
# Ensure CARLA server is running before starting agents
# Check port configurations (default: 2000)
# Verify no firewall blocking connections
```

### Performance Issues

**Low Frame Rate:**
- Reduce image resolution in sensor configuration
- Adjust RAFT-Stereo iteration count
- Enable mixed precision inference

**Memory Issues:**
- Reduce point cloud density parameters
- Limit obstacle list size
- Configure garbage collection intervals

## Building ORB-SLAM3 Examples

### Prerequisites and Dependencies

To successfully build all ORB-SLAM3 examples, you'll need to install several dependencies beyond those listed in the ORB-SLAM3 documentation:

#### Required System Dependencies
```bash
# Update system packages
sudo apt update

# Install basic development tools
sudo apt install -y build-essential cmake git

# Install OpenCV and computer vision libraries
sudo apt install -y libopencv-dev

# Install Eigen3 for mathematical operations
sudo apt install -y libeigen3-dev

# Install OpenGL and graphics libraries
sudo apt install -y libgl1-mesa-dev libglew-dev

# Install ncurses libraries (crucial for examples)
sudo apt install -y libncurses-dev libncursesw5-dev libtinfo-dev

# Install additional dependencies
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev
```

#### Pangolin Installation

Pangolin is required for visualization. Install a compatible version:

```bash
# Install Pangolin v0.6 (compatible version)
cd /tmp
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout v0.6
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-copy -Wno-error" ..
make -j4
sudo make install
sudo ldconfig
```

#### CMakeLists.txt Configuration

The ORB-SLAM3 CMakeLists.txt needs to be updated to properly link all required libraries. The following modifications are automatically applied:

```cmake
# Add OpenGL and GLEW finding
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# Update library linking to include all required libraries
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${OPENGL_LIBRARIES}
${GLEW_LIBRARIES}
-lboost_serialization
-lcrypto
-lncursesw
-ltinfo
-lffi
)
```

### Building Process

1. **Navigate to ORB-SLAM3 directory:**
```bash
cd ORB_SLAM3
```

2. **Run the build script:**
```bash
chmod +x build.sh
./build.sh
```

3. **Or build manually:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Troubleshooting Common Issues

#### 1. Pangolin Not Found Error
```
Could not find required component: Pangolin
```
**Solution:** Install Pangolin v0.6 as shown above. Newer versions may have compatibility issues.

#### 2. OpenGL/GLEW Linking Errors
```
undefined reference to `glXXXX`
```
**Solution:** Install OpenGL and GLEW development packages and ensure they're properly linked in CMakeLists.txt.

#### 3. ncurses Linking Errors
```
undefined reference to `mousemask@NCURSESW6_5.1.20000708`
```
**Solution:** Install ncurses wide character libraries:
```bash
sudo apt install -y libncursesw5-dev libtinfo-dev
```

#### 4. G2O/Eigen Warnings
```
warning: 'Eigen::AlignedBit' is deprecated
```
**Solution:** These are warnings from using newer Eigen versions. They don't affect functionality.

### Built Examples

After successful compilation, you'll have these executables:

#### **Main Library**
- `lib/libORB_SLAM3.so` - Main SLAM library

#### **Monocular SLAM Examples**
- `Examples/Monocular/mono_euroc` - EuRoC dataset monocular SLAM
- `Examples/Monocular/mono_kitti` - KITTI dataset monocular SLAM
- `Examples/Monocular/mono_tum` - TUM dataset monocular SLAM
- `Examples/Monocular/mono_tum_vi` - TUM-VI dataset monocular SLAM

#### **Monocular-Inertial SLAM Examples**
- `Examples/Monocular-Inertial/mono_inertial_euroc` - EuRoC dataset with IMU
- `Examples/Monocular-Inertial/mono_inertial_tum_vi` - TUM-VI dataset with IMU

#### **RGB-D SLAM Examples**
- `Examples/RGB-D/rgbd_tum` - TUM RGB-D dataset

#### **Stereo SLAM Examples**
- `Examples/Stereo/stereo_euroc` - EuRoC stereo dataset
- `Examples/Stereo/stereo_euroc_foundationstereo` - EuRoC with FoundationStereo
- `Examples/Stereo/stereo_kitti` - KITTI stereo dataset
- `Examples/Stereo/stereo_tum_vi` - TUM-VI stereo dataset

#### **Stereo-Inertial SLAM Examples**
- `Examples/Stereo-Inertial/stereo_inertial_euroc` - EuRoC stereo with IMU
- `Examples/Stereo-Inertial/stereo_inertial_tum_vi` - TUM-VI stereo with IMU

#### **Testing Examples**
- `Examples/test_foundationstereo_slam` - FoundationStereo testing
- `Examples/test_stereo_disparity` - Stereo disparity testing

### Usage Examples

```bash
# Run monocular SLAM on TUM dataset
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml PATH_TO_SEQUENCE

# Run stereo SLAM on EuRoC dataset
./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data PATH_TO_SEQUENCE/cam1/data Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt

# Run stereo-inertial SLAM
./Examples/Stereo-Inertial/stereo_inertial_euroc Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data PATH_TO_SEQUENCE/cam1/data Examples/Stereo-Inertial/EuRoC_TimeStamps/SEQUENCE.txt Examples/Stereo-Inertial/EuRoC_IMU/SEQUENCE.txt
```

### Performance Notes

- **Build time:** ~10-15 minutes on a modern system
- **Storage:** ~500MB for complete build
- **Memory:** 4GB+ RAM recommended during compilation
- **Dependencies:** All resolved through standard Ubuntu repositories

## Development

### Working with Submodules

This project includes ORB-SLAM3 as a git submodule. Here's how to work with it:

**Updating the submodule to latest changes:**
```bash
cd ORB_SLAM3
git pull origin main  # or master, depending on default branch
cd ..
git add ORB_SLAM3
git commit -m "Update ORB_SLAM3 submodule"
```

**Making changes to ORB-SLAM3:**
```bash
# Work inside the submodule
cd ORB_SLAM3
# Make your changes, commit them
git add .
git commit -m "Your changes to ORB_SLAM3"
git push origin main  # Push to your fork

# Update parent repository to track the new commit
cd ..
git add ORB_SLAM3
git commit -m "Update ORB_SLAM3 submodule to latest changes"
```

**Switching ORB-SLAM3 to a different branch:**
```bash
cd ORB_SLAM3
git checkout feature-branch
cd ..
git add ORB_SLAM3
git commit -m "Switch ORB_SLAM3 to feature-branch"
```

### Adding New Agents

1. Create new agent class inheriting from `AutonomousAgent`
2. Implement required methods: `setup()`, `sensors()`, `run_step()`
3. Add entry point function for leaderboard integration
4. Test with simulation environment

### Extending Functionality

- **Sensor Integration**: Add new sensor types in `sensors()` method
- **Vision Algorithms**: Implement new depth estimation techniques
- **Navigation**: Develop advanced path planning algorithms
- **ROS Integration**: Add additional publishers/subscribers

## Results and Evaluation

Results are automatically saved to:
- `results/`: Mission logs and performance metrics
- `captured_images/`: Sensor data recordings
- Point cloud files: Combined 3D reconstructions

### Metrics Tracked
- Navigation accuracy and efficiency
- Obstacle detection performance
- Path planning effectiveness
- Computational resource utilization

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{lunar-autonomy-challenge,
  title={Lunar Autonomy Challenge: Advanced Stereo Vision Navigation},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/[your-repo]/LunarAutonomyChallenge}}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-agent`)
3. Commit changes (`git commit -am 'Add new navigation agent'`)
4. Push to branch (`git push origin feature/new-agent`)
5. Create Pull Request

## Support

For questions and support:
- Check existing issues in the repository
- Create new issue with detailed description
- Include logs and configuration details
- Specify simulation environment and hardware setup 