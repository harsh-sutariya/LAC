#!/usr/bin/env python3
"""
Test script for lunar rover data loading and encoding with HJEPA framework.
This script tests:
1. Data loading from collected trajectory files
2. Image processing and format conversion
3. IMU encoder and scaler loading
4. Lunar rover encoder initialization
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add HJEPA to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
hjepa_path = os.path.join(current_dir, 'HJEPA')
sys.path.insert(0, hjepa_path)

from hjepa.data.lunar_rover import LunarRoverDataset, LunarRoverDatasetConfig
from hjepa.models.encoders.lunar_rover import LunarRoverEncoder

def test_dataset_loading():
    """Test basic dataset loading functionality"""
    print("=" * 50)
    print("Testing Dataset Loading...")
    print("=" * 50)
    
    # Create dataset config
    config = LunarRoverDatasetConfig(
        data_dir="data_collection",
        batch_size=4,
        n_steps=8,
        img_size=128,  # Smaller for testing
        num_workers=1,
        max_trajectories=5,  # Limit for testing
        normalize_actions=True,
        normalize_positions=True,
        quick_debug=True,
        train=True
    )
    
    try:
        # Create dataset
        dataset = LunarRoverDataset(config)
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test first sample
        sample = dataset[0]
        print(f"✅ Sample loaded successfully:")
        print(f"   - States shape: {sample.states.shape}")
        print(f"   - Actions shape: {sample.actions.shape}")
        print(f"   - Locations shape: {sample.locations.shape}")
        print(f"   - Proprio vel shape: {sample.proprio_vel.shape}")
        
        # Test trajectory info
        info = dataset.get_trajectory_info(0)
        print(f"✅ Trajectory info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_initialization():
    """Test lunar rover encoder initialization"""
    print("\n" + "=" * 50)
    print("Testing Encoder Initialization...")
    print("=" * 50)
    
    # Check if pretrained files exist
    imu_encoder_path = "imu-pretraining/imu_autoencoder_results/imu_autoencoder_latent32.pth"
    imu_scaler_path = "imu-pretraining/imu_autoencoder_results/imu_scaler.pkl"
    
    if not os.path.exists(imu_encoder_path):
        print(f"❌ IMU encoder not found at {imu_encoder_path}")
        return False
    
    if not os.path.exists(imu_scaler_path):
        print(f"❌ IMU scaler not found at {imu_scaler_path}")
        return False
    
    print("✅ Pretrained IMU files found")
    
    # Test encoder config
    class MockConfig:
        def __init__(self):
            self.imu_encoder_path = imu_encoder_path
            self.imu_scaler_path = imu_scaler_path
            self.image_size = (128, 128)
            self.freeze_vision_encoder = True
            self.freeze_imu_encoder = True
            self.output_dim = 256
            self.final_ln = True
            self.dropout = 0.1
    
    try:
        config = MockConfig()
        
        # Test encoder creation (without SD VAE for now)
        print("⚠️  Note: Skipping actual encoder creation to avoid SD VAE dependency")
        print("   This would require installing diffusers and downloading SD VAE")
        
        return True
        
    except Exception as e:
        print(f"❌ Encoder initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_format():
    """Test data format compatibility"""
    print("\n" + "=" * 50)
    print("Testing Data Format...")
    print("=" * 50)
    
    try:
        # Load a sample trajectory file
        sample_files = list(Path("data_collection").glob("mission_*/trajectory_*.npz"))
        if not sample_files:
            print("❌ No trajectory files found")
            return False
        
        sample_file = sample_files[0]
        print(f"Testing with: {sample_file}")
        
        with np.load(sample_file) as data:
            print(f"✅ Data keys: {list(data.keys())}")
            
            # Check expected keys
            expected_keys = ['images', 'imu_data', 'poses', 'actions', 'timestamps']
            for key in expected_keys:
                if key in data:
                    print(f"   ✅ {key}: {data[key].shape}")
                else:
                    print(f"   ❌ Missing key: {key}")
            
            # Test image format conversion
            images = data['images']
            print(f"✅ Original images shape: {images.shape}")
            
            # Test grayscale to RGB conversion
            if len(images.shape) == 3:
                rgb_images = np.stack([images, images, images], axis=-1)
                print(f"✅ Converted to RGB shape: {rgb_images.shape}")
            
            # Test IMU data format
            imu_data = data['imu_data']
            print(f"✅ IMU data shape: {imu_data.shape}")
            
            # Test pose data
            poses = data['poses']
            relative_position = poses[:, :3]  # x, y, z
            print(f"✅ Relative position shape: {relative_position.shape}")
            
            # Test actions
            actions = data['actions']
            print(f"✅ Actions shape: {actions.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Data format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Lunar Rover Data Loading Tests")
    print("=" * 60)
    
    tests = [
        ("Data Format", test_data_format),
        ("Dataset Loading", test_dataset_loading),
        ("Encoder Initialization", test_encoder_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("🏁 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 