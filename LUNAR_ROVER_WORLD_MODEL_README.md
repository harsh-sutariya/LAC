# Lunar Rover World Model Implementation

This implementation realizes the vision described in `plan.md` - a visual navigation system for lunar rovers using a learned world model with latent dynamics and Model Predictive Control (MPC) planning.

## 🚀 Overview

The system implements a **Joint Embedding Predictive Architecture (JEPA)** that learns to predict future states in latent space given current observations and actions. The world model integrates:

- **RGB Images**: Processed via pretrained Stable Diffusion VAE
- **IMU Data**: Encoded using a pretrained autoencoder (6 DOF × 2 timesteps)
- **Relative Position**: Rover's position relative to start point
- **Actions**: Continuous velocity commands (linear and angular velocity)

## 🏗️ Architecture

### World Model Components

1. **Observation Encoder**: Multi-modal fusion of:
   - Vision: Stable Diffusion VAE (frozen) for image encoding
   - IMU: Pretrained autoencoder for sensor fusion
   - Position: MLP for relative coordinate encoding

2. **Latent Dynamics Model**: 
   - Learns state transitions in latent space: `z_{t+1} = f(z_t, a_t)`
   - Trained with VICReg (Joint Embedding Predictive Architecture)
   - Includes Inverse Dynamics Model (IDM) for action prediction

3. **Planning Module**:
   - Model Predictive Control (MPC) with Cross-Entropy Method (CEM)
   - Plans action sequences to reach goal locations
   - Receding horizon control for real-time navigation

## 📁 Project Structure

```
├── HJEPA/                              # HJEPA framework
│   ├── hjepa/
│   │   ├── configs/lunar_rover/        # Configuration files
│   │   ├── data/lunar_rover.py         # Dataset implementation
│   │   ├── models/encoders/lunar_rover.py  # Encoder architecture
│   │   └── train.py                    # Training framework
├── data_collection/                    # Collected trajectory data
│   ├── mission_1/
│   │   ├── trajectory_0.npz
│   │   └── ...
│   └── collection_summary.json
├── imu-pretraining/                    # Pretrained IMU encoder
│   └── imu_autoencoder_results/
│       ├── imu_autoencoder_latent32.pth
│       └── imu_scaler.pkl
├── train_lunar_rover_world_model.py    # Main training script
├── lunar_rover_navigation.py           # Navigation/planning script
├── test_lunar_rover_data_loading.py    # Testing utilities
└── plan.md                             # Original project plan
```

## 🔧 Installation & Setup

### Prerequisites

```bash
# Install required packages
pip install torch torchvision diffusers scikit-learn omegaconf wandb tqdm matplotlib

# OR install from HJEPA requirements
cd HJEPA
pip install -r requirements.txt
```

### Data Requirements

1. **Collected Trajectory Data**: Use `RunContinuousDataCollection.sh` to collect data
2. **Pretrained IMU Encoder**: Use `imu-pretraining/run_imu_autoencoder.sh` to train

## 🎯 Usage

### 1. Data Collection

```bash
# Collect trajectory data (if not already done)
./RunContinuousDataCollection.sh
```

### 2. IMU Encoder Training

```bash
# Train IMU autoencoder (if not already done)
cd imu-pretraining
./run_imu_autoencoder.sh
```

### 3. World Model Training

```bash
# Train the world model
python3 train_lunar_rover_world_model.py

# Debug mode (smaller model, fewer epochs)
python3 train_lunar_rover_world_model.py --debug

# Custom configuration
python3 train_lunar_rover_world_model.py --values epochs=100 base_lr=0.0005
```

### 4. Navigation/Planning

```bash
# Navigate to a goal location
python3 lunar_rover_navigation.py \
    --model_path checkpoints/lunar_rover_world_model/model.pth \
    --goal_x 5.0 --goal_y 5.0 \
    --start_x 0.0 --start_y 0.0

# Custom planning parameters
python3 lunar_rover_navigation.py \
    --model_path checkpoints/lunar_rover_world_model/model.pth \
    --goal_x 10.0 --goal_y -5.0 \
    --horizon 30 --max_steps 100
```

## 📊 Data Format

### Trajectory Files (.npz)

Each trajectory file contains:
- `images`: Grayscale images (T, H, W) - converted to RGB internally
- `imu_data`: IMU readings (T, 12) - 6 DOF × 2 timesteps
- `poses`: 6 DOF poses (T, 6) - [x, y, z, roll, pitch, yaw]
- `actions`: Velocity commands (T, 2) - [linear_vel, angular_vel]
- `timestamps`: Timestamps for each frame

### Model Input Format

- **States**: RGB images (B, 3, H, W) normalized to [0, 1]
- **Actions**: Velocity commands (B, 2) normalized
- **Locations**: Relative positions (B, 3) normalized
- **Proprio**: IMU data (B, 12) scaled using pretrained scaler

## 🧠 Training Objectives

### VICReg Loss
Joint Embedding Predictive Architecture with:
- **Similarity**: Encourages similar states to have similar representations
- **Variance**: Prevents representation collapse
- **Covariance**: Decorrelates representation dimensions

### Inverse Dynamics Model (IDM)
Predicts actions given current and next states:
- Helps learn action-relevant representations
- Improves planning by understanding action effects

## 🎛️ Configuration

### Main Configuration (`lunar_rover_base.yaml`)

```yaml
# Sequence length
n_steps: 16

# Architecture
hjepa:
  level1:
    backbone:
      arch: lunar_rover
      imu_encoder_path: "../imu-pretraining/imu_autoencoder_results/imu_autoencoder_latent32.pth"
      output_dim: 512
    action_dim: 2

# Training objectives
objectives_l1:
  objectives: [VICReg, IDM]
  
# Optimization
optimizer_type: Adam
base_lr: 0.001
epochs: 50
```

### Debug Configuration (`lunar_rover_debug.yaml`)

Smaller model for faster testing:
- Reduced image size (128×128)
- Fewer epochs (3)
- Smaller batch size (4)
- Limited trajectories (5)

## 🔍 Key Features

### 1. Multi-Modal Sensor Fusion
- Combines vision, IMU, and position information
- Pretrained encoders for each modality
- Learned fusion in latent space

### 2. Latent Dynamics Learning
- Learns environment dynamics in compressed latent space
- Enables efficient planning without pixel-level simulation
- Generalizes to novel goal locations

### 3. Model Predictive Control
- Plans optimal action sequences using learned model
- Cross-Entropy Method for action optimization
- Receding horizon for real-time navigation

### 4. Modular Architecture
- Separate encoding, dynamics, and planning components
- Easy to extend with new modalities or planning algorithms
- Supports hierarchical planning (Level 2 JEPA)

## 📈 Expected Results

### Training Metrics
- **Reconstruction Loss**: Decreases as model learns to predict
- **VICReg Loss**: Balances similarity, variance, and covariance
- **IDM Loss**: Improves action prediction accuracy

### Navigation Performance
- **Success Rate**: Percentage of goals reached
- **Path Efficiency**: Actual vs. optimal path length
- **Planning Time**: Time to compute action sequences

## 🔧 Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Verify trajectory files exist in `data_collection/`
   - Check image format compatibility (grayscale → RGB conversion)

2. **IMU Encoder Issues**:
   - Ensure IMU autoencoder is trained first
   - Check scaler file exists and is compatible

3. **Memory Issues**:
   - Reduce batch size in configuration
   - Use debug mode for testing
   - Enable gradient checkpointing

4. **Training Instability**:
   - Adjust learning rate
   - Check data normalization
   - Verify loss coefficients

## 🚀 Extensions

### Future Enhancements

1. **Hierarchical Planning**: Enable Level 2 JEPA for long-horizon planning
2. **Real Robot Integration**: Add sim-to-real transfer capabilities
3. **Obstacle Avoidance**: Integrate collision prediction in planning
4. **Multi-Goal Navigation**: Plan trajectories through multiple waypoints
5. **Uncertainty Estimation**: Add probabilistic planning for robust navigation

### Advanced Features

1. **Trajectory Stitching**: Combine learned motion primitives
2. **Online Learning**: Adapt model during deployment
3. **Multi-Scale Planning**: Combine local and global planning
4. **Semantic Goals**: Navigate to semantic targets (e.g., "large rock")

## 📚 References

This implementation is based on:
- **Planning with Latent Dynamics Model (PLDM)** - Sobal et al.
- **Navigation World Models** - Bar et al.
- **Joint Embedding Predictive Architecture (JEPA)** - LeCun et al.
- **Model Predictive Control** - Cross-Entropy Method for action optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration files
3. Test with debug mode
4. Submit detailed issue reports

**Happy Navigating! 🌙🚀** 