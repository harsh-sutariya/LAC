import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
from typing import Optional, Tuple

from hjepa.models.encoders.base_class import SequenceBackbone
from hjepa.models.encoders.enums import BackboneOutput
from hjepa.models.misc import build_mlp

# Import for Stable Diffusion VAE
try:
    from diffusers import AutoencoderKL
except ImportError:
    print("Warning: diffusers not installed. Install with: pip install diffusers")
    AutoencoderKL = None


class IMUAutoencoder(nn.Module):
    """IMU Autoencoder from the pretrained model"""
    
    def __init__(self, input_dim=12, latent_dim=32, l2_reg=1e-4):
        super(IMUAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l2_reg = l2_reg
        
        # Encoder: Input(12) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: Input(latent_dim) → Dense(64) → ReLU → Dense(128) → ReLU → Dense(12)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class LunarRoverEncoder(SequenceBackbone):
    """
    Lunar Rover Encoder that combines:
    - RGB images via pretrained Stable Diffusion VAE
    - IMU data via pretrained IMU autoencoder  
    - Relative position via MLP
    """
    
    def __init__(
        self,
        config,
        input_dim,  # Not used directly, but kept for interface compatibility
        input_proprio_dim: int = 0,  # IMU dim (12)
        input_loc_dim: int = 0,      # Position dim (2 or 3)
        normalizer=None,
        imu_encoder_path: Optional[str] = None,
        imu_scaler_path: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        freeze_vision_encoder: bool = True,
        freeze_imu_encoder: bool = True,
    ):
        super().__init__()
        
        self.config = config
        self.normalizer = normalizer
        self.image_size = image_size
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_imu_encoder = freeze_imu_encoder
        
        # Initialize vision encoder (Stable Diffusion VAE)
        if AutoencoderKL is None:
            raise ImportError("diffusers not installed. Run: pip install diffusers")
        
        print("Loading Stable Diffusion VAE...")
        self.vision_encoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        
        if self.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        
        # Vision encoder output: latent shape is (C, H/8, W/8) where C=4 for SD VAE
        # For 256x256 input -> 4x32x32 = 4096 features
        vision_latent_dim = 4 * (image_size[0] // 8) * (image_size[1] // 8)
        
        # Initialize IMU encoder
        self.imu_encoder = None
        self.imu_scaler = None
        if input_proprio_dim > 0:
            self.imu_encoder = IMUAutoencoder(input_dim=input_proprio_dim, latent_dim=32)
            
            # Load pretrained weights if provided
            if imu_encoder_path:
                print(f"Loading pretrained IMU encoder from {imu_encoder_path}")
                checkpoint = torch.load(imu_encoder_path, map_location='cpu')
                self.imu_encoder.load_state_dict(checkpoint)
            
            # Load scaler if provided
            if imu_scaler_path:
                print(f"Loading IMU scaler from {imu_scaler_path}")
                with open(imu_scaler_path, 'rb') as f:
                    self.imu_scaler = pickle.load(f)
            
            if self.freeze_imu_encoder:
                for param in self.imu_encoder.parameters():
                    param.requires_grad = False
                self.imu_encoder.eval()
        
        # Position encoder (simple MLP)
        self.position_encoder = None
        position_latent_dim = 0
        if input_loc_dim > 0:
            position_latent_dim = 32
            self.position_encoder = build_mlp(
                layers_dims=[input_loc_dim, 64, 64, position_latent_dim],
                activation='relu'
            )
        
        # Fusion layer to combine all modalities
        total_input_dim = vision_latent_dim
        if self.imu_encoder:
            total_input_dim += 32  # IMU latent dim
        if self.position_encoder:
            total_input_dim += position_latent_dim
        
        # Output projection
        self.output_dim = getattr(config, 'output_dim', 512)
        self.fusion = build_mlp(
            layers_dims=[total_input_dim, 1024, 512, self.output_dim],
            activation='relu',
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # Layer norm
        self.final_ln = nn.LayerNorm(self.output_dim) if getattr(config, 'final_ln', True) else nn.Identity()
        
        # Store dimensions for HJEPA
        self.using_proprio = input_proprio_dim > 0
        self.using_location = input_loc_dim > 0
        self.output_obs_dim = self.output_dim
        self.output_proprio_dim = 0  # We fuse everything, so no separate proprio output
        self.output_loc_dim = 0     # We fuse everything, so no separate location output
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for Stable Diffusion VAE
        Expected input: (B, C, H, W) with values in [0, 1]
        """
        # Ensure image is in correct format and range
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Resize if needed
        if image.shape[-2:] != self.image_size:
            image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False)
        
        # Convert to [-1, 1] range for SD VAE
        image = 2.0 * image - 1.0
        
        return image
    
    def preprocess_imu(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess IMU data using the trained scaler
        """
        if self.imu_scaler is not None:
            # Convert to numpy for scaler, then back to torch
            device = imu_data.device
            imu_np = imu_data.detach().cpu().numpy()
            imu_scaled = self.imu_scaler.transform(imu_np)
            imu_data = torch.from_numpy(imu_scaled).float().to(device)
        
        return imu_data
    
    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using Stable Diffusion VAE"""
        image = self.preprocess_image(image)
        
        # Encode with SD VAE
        with torch.no_grad() if self.freeze_vision_encoder else torch.enable_grad():
            latent_dist = self.vision_encoder.encode(image).latent_dist
            # Use mode instead of sampling for deterministic behavior
            latent = latent_dist.mode()
            # Apply scaling factor used in SD
            latent = latent * 0.18215
        
        # Flatten spatial dimensions
        B = latent.shape[0]
        latent = latent.view(B, -1)
        
        return latent
    
    def encode_imu(self, imu_data: torch.Tensor) -> torch.Tensor:
        """Encode IMU data using pretrained autoencoder"""
        if self.imu_encoder is None:
            raise ValueError("IMU encoder not initialized")
        
        imu_data = self.preprocess_imu(imu_data)
        
        with torch.no_grad() if self.freeze_imu_encoder else torch.enable_grad():
            latent = self.imu_encoder.encode(imu_data)
        
        return latent
    
    def encode_position(self, position: torch.Tensor) -> torch.Tensor:
        """Encode relative position"""
        if self.position_encoder is None:
            raise ValueError("Position encoder not initialized")
        return self.position_encoder(position)
    
    def forward(self, x, proprio=None, location=None, **kwargs):
        """
        Forward pass
        
        Args:
            x: RGB image tensor (B, C, H, W)
            proprio: IMU data tensor (B, 12) - optional
            location: Relative position tensor (B, 2 or 3) - optional
        """
        features = []
        
        # Encode vision
        vision_features = self.encode_vision(x)
        features.append(vision_features)
        
        # Encode IMU if available
        if proprio is not None and self.imu_encoder is not None:
            imu_features = self.encode_imu(proprio)
            features.append(imu_features)
        
        # Encode position if available
        if location is not None and self.position_encoder is not None:
            position_features = self.encode_position(location)
            features.append(position_features)
        
        # Fuse all features
        fused_features = torch.cat(features, dim=-1)
        output = self.fusion(fused_features)
        output = self.final_ln(output)
        
        return BackboneOutput(encodings=output)
    
    def forward_multiple(self, x, proprio=None, location=None, **kwargs):
        """Forward pass for multiple timesteps (for sequence processing)"""
        if x.dim() == 5:  # (T, B, C, H, W)
            T, B = x.shape[:2]
            # Reshape to (T*B, C, H, W)
            x = x.view(T*B, *x.shape[2:])
            if proprio is not None:
                proprio = proprio.view(T*B, *proprio.shape[2:])
            if location is not None:
                location = location.view(T*B, *location.shape[2:])
            
            # Forward pass
            result = self.forward(x, proprio=proprio, location=location, **kwargs)
            
            # Reshape back to (T, B, D)
            encodings = result.encodings.view(T, B, -1)
            return BackboneOutput(encodings=encodings)
        else:
            return self.forward(x, proprio=proprio, location=location, **kwargs)


def build_lunar_rover_encoder(config, **kwargs):
    """Factory function to build lunar rover encoder"""
    return LunarRoverEncoder(config, **kwargs) 