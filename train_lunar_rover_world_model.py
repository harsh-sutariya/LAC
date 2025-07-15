#!/usr/bin/env python3
"""
Training script for Lunar Rover World Model using HJEPA framework.

This script implements the plan.md by:
1. Loading collected lunar rover data (images, IMU, poses, actions)
2. Training a world model using Joint Embedding Predictive Architecture (JEPA)
3. Learning latent dynamics with VICReg and IDM objectives
4. Saving the trained model for later use in planning/navigation

Usage:
    python3 train_lunar_rover_world_model.py [--config CONFIG_PATH] [--debug]
"""

import sys
import os
import argparse
from pathlib import Path

# Add HJEPA to Python path
sys.path.insert(0, 'HJEPA')

from hjepa.train import Trainer
from hjepa.configs import load_config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Lunar Rover World Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="HJEPA/hjepa/configs/lunar_rover/lunar_rover_base.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Use debug configuration with smaller model and fewer epochs"
    )
    parser.add_argument(
        "--values", 
        nargs="+", 
        help="Override config values (e.g., --values epochs=10 batch_size=16)"
    )
    
    args = parser.parse_args()
    
    # Use debug config if requested
    if args.debug:
        config_path = "HJEPA/hjepa/configs/lunar_rover/lunar_rover_debug.yaml"
    else:
        config_path = args.config
    
    print("🚀 Starting Lunar Rover World Model Training")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Debug mode: {args.debug}")
    
    # Check if data directory exists
    data_dir = Path("data_collection")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure you have collected data first using RunContinuousDataCollection.sh")
        return 1
    
    # Count available trajectories
    trajectory_files = list(data_dir.glob("mission_*/trajectory_*.npz"))
    print(f"📊 Found {len(trajectory_files)} trajectory files")
    
    if len(trajectory_files) == 0:
        print("❌ No trajectory files found in data directory")
        return 1
    
    # Check if IMU encoder exists
    imu_encoder_path = "imu-pretraining/imu_autoencoder_results/imu_autoencoder_latent32.pth"
    if not Path(imu_encoder_path).exists():
        print(f"❌ IMU encoder not found: {imu_encoder_path}")
        print("Please train the IMU encoder first using run_imu_autoencoder.sh")
        return 1
    
    print("✅ All required files found")
    
    try:
        # Load configuration
        print(f"📝 Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Apply command line value overrides
        if args.values:
            print("🔧 Applying configuration overrides:")
            for value_override in args.values:
                if "=" in value_override:
                    key, value = value_override.split("=", 1)
                    print(f"   {key} = {value}")
                    # Simple override (can be extended for nested keys)
                    if hasattr(config, key):
                        # Try to infer type
                        try:
                            if value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            elif '.' in value:
                                value = float(value)
                        except:
                            pass  # Keep as string
                        setattr(config, key, value)
        
        # Create trainer
        print("🏗️  Initializing trainer...")
        trainer = Trainer(config)
        
        # Display model information
        print("🧠 Model Information:")
        print(f"   - Parameters: {trainer.n_parameters:,}")
        print(f"   - Training objectives: {config.objectives_l1.objectives}")
        print(f"   - Epochs: {config.epochs}")
        print(f"   - Learning rate: {config.base_lr}")
        
        # Start training
        print("\n🎯 Starting training...")
        print("=" * 60)
        
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
        
        # Display final results
        output_dir = Path(config.output_root) / config.output_dir
        print(f"📁 Model saved to: {output_dir}")
        print(f"📊 Training logs: {output_dir}/logs")
        
        # Next steps
        print("\n🚀 Next Steps:")
        print("1. You can now use the trained model for planning and navigation")
        print("2. Run evaluation to test the learned representations")
        print("3. Implement MPC planning for goal-conditioned navigation")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 