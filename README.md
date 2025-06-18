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
├── models/                     # Pre-trained neural network weights
├── docs/                       # Documentation and fiducials
├── docker/                     # Docker configuration files
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Create Conda Environment

Set up the required Python environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate lunar-autonomy
```

### 2. Launch Lunar Simulator

Start the CARLA-based lunar simulation environment:

```bash
bash RunLunarSimulator.sh
```

This will:
- Initialize the CARLA simulation server
- Load the lunar surface environment
- Set up sensor configurations
- Prepare the rover for autonomous navigation

### 3. Run Leaderboard Evaluation

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

## Development

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