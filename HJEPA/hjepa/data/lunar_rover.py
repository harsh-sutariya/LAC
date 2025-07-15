from typing import NamedTuple, Optional, List
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from omegaconf import MISSING
import pickle
import os
from PIL import Image


class LunarRoverSample(NamedTuple):
    """Data sample for lunar rover navigation"""
    states: torch.Tensor  # RGB images: [(batch_size), T, C, H, W]
    locations: torch.Tensor  # Relative position: [(batch_size), T, 2 or 3]
    actions: torch.Tensor  # Velocity commands: [(batch_size), T-1, action_dim]
    indices: torch.Tensor  # [(batch_size)] for prioritized replay
    proprio_pos: torch.Tensor  # Empty tensor (not used)
    proprio_vel: torch.Tensor  # IMU data: [batch_size, T, 12]
    # For hierarchy (not used initially)
    l2_states: torch.Tensor
    l2_locations: torch.Tensor
    l2_proprio_vel: torch.Tensor
    l2_proprio_pos: torch.Tensor


@dataclass
class LunarRoverDatasetConfig:
    """Configuration for lunar rover dataset"""
    data_dir: str = MISSING  # Path to data collection directory
    num_workers: int = 4
    batch_size: int = 32
    seed: int = 0
    quick_debug: bool = False
    val_fraction: float = 0.2
    train: bool = True
    n_steps: int = 16  # Sequence length
    l2_n_steps: int = 0  # For hierarchical models (not used initially)
    l2_step_skip: int = 4
    
    # Image settings
    img_size: int = 256
    image_format: str = "RGB"  # RGB or BGR
    
    # Data loading options
    max_trajectories: Optional[int] = None  # Limit number of trajectories
    trajectory_subsample: int = 1  # Subsample every N frames
    normalize_actions: bool = True
    normalize_positions: bool = True
    
    # Augmentation options
    random_crop: bool = False
    random_flip: bool = False
    color_jitter: bool = False
    
    def __post_init__(self):
        # Ensure data directory exists
        if self.data_dir != MISSING and not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")


class LunarRoverDataset(torch.utils.data.Dataset):
    """
    Dataset for lunar rover navigation data
    
    Expected data structure:
    data_dir/
    ├── mission_1/
    │   ├── trajectory_000.npz
    │   ├── trajectory_001.npz
    │   └── ...
    ├── mission_2/
    │   └── ...
    └── ...
    
    Each trajectory file contains:
    - rgb_images: (T, H, W, 3) RGB images
    - imu_data: (T, 12) IMU readings (2 timesteps × 6 features)
    - relative_position: (T, 2) or (T, 3) relative position from start
    - actions: (T-1, action_dim) velocity commands
    - timestamps: (T,) timestamp for each frame
    """
    
    def __init__(self, config: LunarRoverDatasetConfig, normalizer=None):
        self.config = config
        self.normalizer = normalizer
        
        # Load trajectory files
        self.trajectory_files = self._find_trajectory_files()
        if self.config.max_trajectories:
            self.trajectory_files = self.trajectory_files[:self.config.max_trajectories]
        
        print(f"Found {len(self.trajectory_files)} trajectory files")
        
        # Load and prepare data
        self.trajectories = self._load_trajectories()
        
        # Create index mapping for efficient access
        self._create_index_mapping()
        
        # Compute normalization statistics if needed
        if self.normalizer is None:
            self._compute_normalization()
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _find_trajectory_files(self) -> List[Path]:
        """Find all trajectory files in the data directory"""
        data_dir = Path(self.config.data_dir)
        trajectory_files = []
        
        # Look for .npz files in mission directories
        for mission_dir in data_dir.iterdir():
            if mission_dir.is_dir() and mission_dir.name.startswith('mission_'):
                for traj_file in mission_dir.glob('trajectory_*.npz'):
                    trajectory_files.append(traj_file)
        
        # Also look for trajectory files directly in data_dir
        for traj_file in data_dir.glob('trajectory_*.npz'):
            trajectory_files.append(traj_file)
        
        trajectory_files.sort()
        return trajectory_files
    
    def _load_trajectories(self) -> List[dict]:
        """Load trajectory data from files"""
        trajectories = []
        
        for traj_file in self.trajectory_files:
            try:
                with np.load(traj_file) as data:
                    # Handle actual data format from collected data
                    # Expected keys: images, imu_data, poses, actions, timestamps
                    
                    # Convert grayscale images to RGB format
                    images = data['images']  # Shape: (T, H, W)
                    if len(images.shape) == 3:
                        # Convert grayscale to RGB by replicating the channel
                        rgb_images = np.stack([images, images, images], axis=-1)  # (T, H, W, 3)
                    else:
                        rgb_images = images  # Already RGB
                    
                    # Extract relative position from poses (first 2 or 3 components)
                    poses = data['poses']  # Shape: (T, 6) [x, y, z, roll, pitch, yaw]
                    relative_position = poses[:, :3]  # Take x, y, z position
                    
                    traj_data = {
                        'rgb_images': rgb_images,
                        'imu_data': data['imu_data'],
                        'relative_position': relative_position,
                        'actions': data['actions'],
                        'timestamps': data.get('timestamps', None),
                        'file_path': str(traj_file)
                    }
                    
                    # Subsample if requested
                    if self.config.trajectory_subsample > 1:
                        step = self.config.trajectory_subsample
                        traj_data['rgb_images'] = traj_data['rgb_images'][::step]
                        traj_data['imu_data'] = traj_data['imu_data'][::step]
                        traj_data['relative_position'] = traj_data['relative_position'][::step]
                        # Actions need special handling (T-1 length)
                        traj_data['actions'] = traj_data['actions'][::step]
                        if traj_data['timestamps'] is not None:
                            traj_data['timestamps'] = traj_data['timestamps'][::step]
                    
                    trajectories.append(traj_data)
                    
            except Exception as e:
                print(f"Error loading {traj_file}: {e}")
                continue
        
        return trajectories
    
    def _create_index_mapping(self):
        """Create mapping from dataset index to trajectory and start frame"""
        self.index_mapping = []
        
        for traj_idx, traj_data in enumerate(self.trajectories):
            traj_length = len(traj_data['rgb_images'])
            max_start_idx = traj_length - self.config.n_steps
            
            for start_idx in range(max(1, max_start_idx + 1)):
                self.index_mapping.append((traj_idx, start_idx))
    
    def _compute_normalization(self):
        """Compute normalization statistics for actions and positions"""
        all_actions = []
        all_positions = []
        
        for traj_data in self.trajectories:
            all_actions.append(traj_data['actions'])
            all_positions.append(traj_data['relative_position'])
        
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0) + 1e-8
        else:
            self.action_mean = np.zeros(2)  # Default for [linear_vel, angular_vel]
            self.action_std = np.ones(2)
        
        if all_positions:
            all_positions = np.concatenate(all_positions, axis=0)
            self.position_mean = np.mean(all_positions, axis=0)
            self.position_std = np.std(all_positions, axis=0) + 1e-8
        else:
            self.position_mean = np.zeros(2)  # Default for [x, y]
            self.position_std = np.ones(2)
    
    def __len__(self) -> int:
        if self.config.quick_debug:
            return min(100, len(self.index_mapping))
        return len(self.index_mapping)
    
    def _normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """Normalize image to [0, 1] range and convert to tensor"""
        # Ensure image is in [0, 255] range and convert to float
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor and rearrange dimensions
        # Input: (H, W, C) -> Output: (C, H, W)
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            # Handle grayscale case (H, W) -> (1, H, W)
            image_tensor = torch.from_numpy(image).unsqueeze(0)
            # Convert to 3-channel by repeating
            image_tensor = image_tensor.repeat(3, 1, 1)
        
        # Resize if needed
        if image_tensor.shape[-2:] != (self.config.img_size, self.config.img_size):
            import torch.nn.functional as F
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.config.img_size, self.config.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return image_tensor
    
    def _normalize_actions(self, actions: np.ndarray) -> torch.Tensor:
        """Normalize actions"""
        if self.config.normalize_actions:
            actions = (actions - self.action_mean) / self.action_std
        return torch.from_numpy(actions).float()
    
    def _normalize_positions(self, positions: np.ndarray) -> torch.Tensor:
        """Normalize positions"""
        if self.config.normalize_positions:
            positions = (positions - self.position_mean) / self.position_std
        return torch.from_numpy(positions).float()
    
    def __getitem__(self, idx: int) -> LunarRoverSample:
        """Get a sequence sample"""
        traj_idx, start_idx = self.index_mapping[idx]
        traj_data = self.trajectories[traj_idx]
        
        end_idx = start_idx + self.config.n_steps
        
        # Extract sequence data
        rgb_sequence = traj_data['rgb_images'][start_idx:end_idx]
        imu_sequence = traj_data['imu_data'][start_idx:end_idx]
        position_sequence = traj_data['relative_position'][start_idx:end_idx]
        action_sequence = traj_data['actions'][start_idx:end_idx-1]  # T-1 actions
        
        # Process images
        states = torch.stack([
            self._normalize_image(img) for img in rgb_sequence
        ])  # (T, C, H, W)
        
        # Process other modalities
        locations = self._normalize_positions(position_sequence)  # (T, 2 or 3)
        actions = self._normalize_actions(action_sequence)  # (T-1, action_dim)
        proprio_vel = torch.from_numpy(imu_sequence).float()  # (T, 12)
        
        # Create sample
        sample = LunarRoverSample(
            states=states,
            locations=locations,
            actions=actions,
            indices=torch.tensor([idx], dtype=torch.long),
            proprio_pos=torch.empty(0),  # Not used
            proprio_vel=proprio_vel,
            # L2 hierarchy (not used initially)
            l2_states=torch.empty(0),
            l2_locations=torch.empty(0),
            l2_proprio_vel=torch.empty(0),
            l2_proprio_pos=torch.empty(0),
        )
        
        return sample
    
    def get_trajectory_info(self, idx: int) -> dict:
        """Get information about the trajectory containing the given sample"""
        traj_idx, start_idx = self.index_mapping[idx]
        traj_data = self.trajectories[traj_idx]
        
        return {
            'trajectory_idx': traj_idx,
            'start_idx': start_idx,
            'file_path': traj_data['file_path'],
            'length': len(traj_data['rgb_images'])
        }
    
    def save_normalization_stats(self, path: str):
        """Save normalization statistics"""
        stats = {
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'position_mean': self.position_mean,
            'position_std': self.position_std,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"Normalization stats saved to {path}")
    
    def load_normalization_stats(self, path: str):
        """Load normalization statistics"""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        
        self.action_mean = stats['action_mean']
        self.action_std = stats['action_std']
        self.position_mean = stats['position_mean']
        self.position_std = stats['position_std']
        
        print(f"Normalization stats loaded from {path}")


def create_lunar_rover_dataset(config: LunarRoverDatasetConfig, normalizer=None):
    """Factory function to create lunar rover dataset"""
    return LunarRoverDataset(config, normalizer=normalizer) 