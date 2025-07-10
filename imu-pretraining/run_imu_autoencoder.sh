#!/bin/bash

# IMU Autoencoder Training Script
# This script provides convenient commands for training the IMU autoencoder

# Check if data directory exists
if [ ! -d "../data_collection" ]; then
    echo "❌ Data collection directory not found!"
    echo "Please ensure you have collected IMU data first:"
    echo "  cd .. && ./RunContinuousDataCollection.sh"
    exit 1
fi

# Count available data
trajectory_count=$(find ../data_collection -name "trajectory_*.npz" | wc -l)
echo "📊 Found $trajectory_count trajectory files for training"

if [ $trajectory_count -lt 10 ]; then
    echo "⚠️  Warning: Very few trajectories found. Consider collecting more data."
fi

echo "🚀 Starting IMU Autoencoder Training..."
echo "="*50

# Parse command line arguments for different configurations
case "${1:-default}" in
    "quick")
        echo "🏃 Quick training (small latent dim, fewer epochs)"
        python imu_autoencoder.py \
            --latent_dim 16 \
            --epochs 50 \
            --batch_size 64 \
            --max_trajectories 100
        ;;
    "full")
        echo "🎯 Full training (recommended settings)"
        python imu_autoencoder.py \
            --latent_dim 32 \
            --epochs 150 \
            --batch_size 128 \
            --learning_rate 1e-3 \
            --weight_decay 1e-4 \
            --l2_reg 1e-4
        ;;
    "large")
        echo "🔥 Large model training (bigger latent space)"
        python imu_autoencoder.py \
            --latent_dim 64 \
            --epochs 200 \
            --batch_size 128 \
            --learning_rate 1e-3
        ;;
    "small")
        echo "⚡ Small model training (compact representation)"
        python imu_autoencoder.py \
            --latent_dim 8 \
            --epochs 100 \
            --batch_size 64 \
            --learning_rate 1e-3
        ;;
    "custom")
        echo "🛠️  Custom training with user parameters"
        echo "Usage: $0 custom [additional arguments]"
        echo "Example: $0 custom --latent_dim 24 --epochs 120"
        shift
        python imu_autoencoder.py "$@"
        ;;
    "help")
        echo "🆘 IMU Autoencoder Training Options:"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  quick   - Fast training for testing (16D latent, 50 epochs)"
        echo "  full    - Recommended settings (32D latent, 150 epochs) [DEFAULT]"
        echo "  large   - Large model (64D latent, 200 epochs)"
        echo "  small   - Compact model (8D latent, 100 epochs)"
        echo "  custom  - Custom parameters (pass additional args)"
        echo "  help    - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 quick                    # Fast training"
        echo "  $0 full                     # Recommended training"
        echo "  $0 custom --latent_dim 24   # Custom latent dimension"
        echo ""
        echo "For full parameter list:"
        echo "  python imu_autoencoder.py --help"
        exit 0
        ;;
    *)
        echo "🎯 Default training (recommended settings)"
        python imu_autoencoder.py \
            --latent_dim 32 \
            --epochs 150 \
            --batch_size 128 \
            --learning_rate 1e-3 \
            --weight_decay 1e-4 \
            --l2_reg 1e-4
        ;;
esac

echo ""
echo "✅ Training completed!"
echo "📁 Results saved to: imu_autoencoder_results/"
echo "📊 Check the following files:"
echo "   - imu_autoencoder_results/training_curves.png"
echo "   - imu_autoencoder_results/feature_reconstruction.png"
echo "   - imu_autoencoder_results/latent_space.png"
echo "   - imu_autoencoder_results/imu_autoencoder_latent*.pth (model)"
echo "   - imu_autoencoder_results/evaluation_metrics.json" 