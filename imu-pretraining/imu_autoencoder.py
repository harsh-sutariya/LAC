#!/usr/bin/env python3
"""
IMU Data Autoencoder for Self-Supervised Representation Learning

This script implements an autoencoder tailored for dual-timestep IMU data sequences
from the lunar rover data collection. The model learns compressed representations
of IMU patterns through reconstruction.

Architecture:
- Input: 12 features (2 timesteps × 6 IMU features each)
- Encoder: 12 → 128 → 64 → latent_dim (with ReLU activations)
- Decoder: latent_dim → 64 → 128 → 12 (with ReLU activations)
- Loss: MSE reconstruction loss
- Regularization: L2 penalty on latent representation

Usage:
    conda activate lunar
    python imu_autoencoder.py --latent_dim 32 --epochs 150
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class IMUDataset(Dataset):
    """Dataset class for IMU data sequences"""
    
    def __init__(self, imu_data, scaler=None, fit_scaler=True):
        """
        Initialize IMU dataset
        
        Args:
            imu_data: numpy array of shape (n_samples, 12)
            scaler: sklearn StandardScaler (optional)
            fit_scaler: whether to fit the scaler on this data
        """
        self.raw_data = imu_data.astype(np.float32)
        
        # Initialize or use provided scaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.raw_data)
        else:
            self.scaler = scaler
            if fit_scaler:
                self.scaler.fit(self.raw_data)
        
        # Normalize data (zero mean, unit variance)
        self.data = self.scaler.transform(self.raw_data).astype(np.float32)
        
        print(f"Dataset initialized: {len(self.data)} samples")
        print(f"Original data range: [{self.raw_data.min():.3f}, {self.raw_data.max():.3f}]")
        print(f"Normalized data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"Normalized data mean: {self.data.mean():.3f}, std: {self.data.std():.3f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return the sample as both input and target for autoencoder
        sample = torch.from_numpy(self.data[idx])
        return sample, sample
    
    def inverse_transform(self, normalized_data):
        """Convert normalized data back to original scale"""
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.detach().cpu().numpy()
        return self.scaler.inverse_transform(normalized_data)

class IMUAutoencoder(nn.Module):
    """MLP-based Autoencoder for IMU data"""
    
    def __init__(self, input_dim=12, latent_dim=32, l2_reg=1e-4):
        """
        Initialize autoencoder
        
        Args:
            input_dim: input feature dimension (12 for dual-timestep IMU)
            latent_dim: latent representation dimension
            l2_reg: L2 regularization weight for latent representation
        """
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
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
    
    def compute_loss(self, x, x_reconstructed, z):
        """
        Compute total loss: reconstruction MSE + L2 regularization on latent
        
        Args:
            x: original input
            x_reconstructed: reconstructed input
            z: latent representation
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
        
        # L2 regularization on latent representation
        l2_loss = self.l2_reg * torch.mean(z ** 2)
        
        total_loss = reconstruction_loss + l2_loss
        
        return total_loss, reconstruction_loss, l2_loss

def load_imu_data_from_missions(data_dir="data_collection", max_trajectories=None):
    """
    Load IMU data from all trajectory files in mission directories
    
    Args:
        data_dir: path to data collection directory
        max_trajectories: maximum number of trajectories to load (None for all)
    
    Returns:
        numpy array of shape (total_samples, 12)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading IMU data from {data_dir}...")
    
    # Find all trajectory files
    trajectory_files = []
    mission_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('mission_')])
    
    for mission_dir in mission_dirs:
        mission_trajectories = list(mission_dir.glob("trajectory_*.npz"))
        trajectory_files.extend(mission_trajectories)
    
    print(f"Found {len(trajectory_files)} trajectory files in {len(mission_dirs)} missions")
    
    if max_trajectories is not None and len(trajectory_files) > max_trajectories:
        # Randomly sample trajectories for faster iteration during development
        np.random.seed(42)
        trajectory_files = np.random.choice(trajectory_files, max_trajectories, replace=False)
        print(f"Sampling {max_trajectories} trajectories for analysis")
    
    # Load IMU data from all trajectories
    all_imu_data = []
    failed_loads = 0
    
    for traj_file in tqdm(trajectory_files, desc="Loading IMU data"):
        try:
            data = np.load(traj_file)
            imu_data = data['imu_data']  # Shape: (n_steps, 12)
            
            # Validate shape
            if imu_data.shape[1] != 12:
                print(f"Warning: {traj_file} has IMU shape {imu_data.shape}, expected (n, 12)")
                continue
            
            all_imu_data.append(imu_data)
            
        except Exception as e:
            print(f"Error loading {traj_file}: {e}")
            failed_loads += 1
    
    if not all_imu_data:
        raise ValueError("No valid IMU data loaded!")
    
    # Concatenate all IMU data
    combined_imu_data = np.concatenate(all_imu_data, axis=0)
    
    print(f"✅ Successfully loaded IMU data:")
    print(f"   Files loaded: {len(all_imu_data)}/{len(trajectory_files)}")
    print(f"   Failed loads: {failed_loads}")
    print(f"   Total samples: {len(combined_imu_data)}")
    print(f"   Data shape: {combined_imu_data.shape}")
    print(f"   Data range: [{combined_imu_data.min():.3f}, {combined_imu_data.max():.3f}]")
    
    return combined_imu_data

def analyze_imu_data(imu_data):
    """Analyze IMU data characteristics"""
    print("\n📊 IMU Data Analysis:")
    print(f"Shape: {imu_data.shape}")
    
    # Feature names for dual-timestep IMU
    feature_names = [
        'prev_acc_x', 'prev_acc_y', 'prev_acc_z',
        'prev_gyro_x', 'prev_gyro_y', 'prev_gyro_z',
        'curr_acc_x', 'curr_acc_y', 'curr_acc_z',
        'curr_gyro_x', 'curr_gyro_y', 'curr_gyro_z'
    ]
    
    print("\nFeature statistics:")
    for i, name in enumerate(feature_names):
        values = imu_data[:, i]
        print(f"  {name:12s}: mean={values.mean():7.3f}, std={values.std():6.3f}, "
              f"range=[{values.min():7.3f}, {values.max():7.3f}]")
    
    # Analyze accelerometer vs gyroscope ranges
    prev_acc = imu_data[:, 0:3]
    prev_gyro = imu_data[:, 3:6]
    curr_acc = imu_data[:, 6:9]
    curr_gyro = imu_data[:, 9:12]
    
    print(f"\nSensor modality ranges:")
    print(f"  Previous accelerometer: [{prev_acc.min():.3f}, {prev_acc.max():.3f}]")
    print(f"  Previous gyroscope:     [{prev_gyro.min():.3f}, {prev_gyro.max():.3f}]")
    print(f"  Current accelerometer:  [{curr_acc.min():.3f}, {curr_acc.max():.3f}]")
    print(f"  Current gyroscope:      [{curr_gyro.min():.3f}, {curr_gyro.max():.3f}]")

def train_autoencoder(model, train_loader, val_loader, device, args):
    """
    Train the autoencoder with specified hyperparameters
    
    Args:
        model: IMUAutoencoder instance
        train_loader: training data loader
        val_loader: validation data loader
        device: torch device
        args: training arguments
    """
    print(f"\n🚀 Starting training on {device}")
    print(f"Training parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  L2 regularization: {model.l2_reg}")
    
    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.learning_rate, 
                           weight_decay=args.weight_decay)
    
    # Learning rate scheduler: Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_reconstruction_loss': [],
        'train_l2_loss': [],
        'val_loss': [],
        'val_reconstruction_loss': [],
        'val_cosine_similarity': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        train_reconstruction_losses = []
        train_l2_losses = []
        
        for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            batch_x = batch_x.to(device)
            
            # Forward pass
            x_reconstructed, z = model(batch_x)
            total_loss, reconstruction_loss, l2_loss = model.compute_loss(batch_x, x_reconstructed, z)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            train_losses.append(total_loss.item())
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_l2_losses.append(l2_loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_reconstruction_losses = []
        all_cosine_similarities = []
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                
                # Forward pass
                x_reconstructed, z = model(batch_x)
                total_loss, reconstruction_loss, l2_loss = model.compute_loss(batch_x, x_reconstructed, z)
                
                # Record losses
                val_losses.append(total_loss.item())
                val_reconstruction_losses.append(reconstruction_loss.item())
                
                # Compute cosine similarity
                x_np = batch_x.cpu().numpy()
                x_recon_np = x_reconstructed.cpu().numpy()
                
                for i in range(len(x_np)):
                    cos_sim = cosine_similarity(x_np[i:i+1], x_recon_np[i:i+1])[0, 0]
                    all_cosine_similarities.append(cos_sim)
        
        # Update learning rate
        scheduler.step()
        
        # Record epoch statistics
        avg_train_loss = np.mean(train_losses)
        avg_train_recon_loss = np.mean(train_reconstruction_losses)
        avg_train_l2_loss = np.mean(train_l2_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_recon_loss = np.mean(val_reconstruction_losses)
        avg_cosine_sim = np.mean(all_cosine_similarities)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['train_reconstruction_loss'].append(avg_train_recon_loss)
        history['train_l2_loss'].append(avg_train_l2_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_reconstruction_loss'].append(avg_val_recon_loss)
        history['val_cosine_similarity'].append(avg_cosine_sim)
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Cosine Sim: {avg_cosine_sim:.4f}, LR: {current_lr:.2e}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    
    return history

def evaluate_autoencoder(model, test_loader, dataset, device, args):
    """
    Evaluate the trained autoencoder
    
    Args:
        model: trained IMUAutoencoder
        test_loader: test data loader
        dataset: test dataset for inverse transformation
        device: torch device
        args: arguments
    """
    print("\n📊 Evaluating autoencoder...")
    
    model.eval()
    
    # Collect predictions and targets
    all_originals = []
    all_reconstructed = []
    all_latent = []
    reconstruction_mses = []
    cosine_similarities = []
    
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            
            # Forward pass
            x_reconstructed, z = model(batch_x)
            
            # Store results
            all_originals.append(batch_x.cpu().numpy())
            all_reconstructed.append(x_reconstructed.cpu().numpy())
            all_latent.append(z.cpu().numpy())
            
            # Compute per-sample MSE
            mse_per_sample = torch.mean((batch_x - x_reconstructed) ** 2, dim=1)
            reconstruction_mses.extend(mse_per_sample.cpu().numpy())
            
            # Compute per-sample cosine similarity
            for i in range(len(batch_x)):
                orig = batch_x[i:i+1].cpu().numpy()
                recon = x_reconstructed[i:i+1].cpu().numpy()
                cos_sim = cosine_similarity(orig, recon)[0, 0]
                cosine_similarities.append(cos_sim)
    
    # Concatenate all results
    all_originals = np.concatenate(all_originals, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_latent = np.concatenate(all_latent, axis=0)
    
    # Compute overall metrics
    overall_mse = np.mean(reconstruction_mses)
    overall_cosine_sim = np.mean(cosine_similarities)
    
    print(f"\n📈 Evaluation Results:")
    print(f"  Reconstruction MSE: {overall_mse:.6f}")
    print(f"  Cosine Similarity: {overall_cosine_sim:.4f}")
    print(f"  MSE std: {np.std(reconstruction_mses):.6f}")
    print(f"  Cosine similarity std: {np.std(cosine_similarities):.4f}")
    
    # Feature-wise analysis
    feature_names = [
        'prev_acc_x', 'prev_acc_y', 'prev_acc_z',
        'prev_gyro_x', 'prev_gyro_y', 'prev_gyro_z',
        'curr_acc_x', 'curr_acc_y', 'curr_acc_z',
        'curr_gyro_x', 'curr_gyro_y', 'curr_gyro_z'
    ]
    
    print(f"\nFeature-wise reconstruction MSE:")
    for i, name in enumerate(feature_names):
        feature_mse = np.mean((all_originals[:, i] - all_reconstructed[:, i]) ** 2)
        print(f"  {name:12s}: {feature_mse:.6f}")
    
    # Return evaluation data for visualization
    eval_data = {
        'originals': all_originals,
        'reconstructed': all_reconstructed,
        'latent': all_latent,
        'reconstruction_mses': reconstruction_mses,
        'cosine_similarities': cosine_similarities,
        'overall_mse': overall_mse,
        'overall_cosine_sim': overall_cosine_sim
    }
    
    return eval_data

def visualize_results(history, eval_data, args, save_dir="imu_autoencoder_results"):
    """
    Create comprehensive visualizations of training and evaluation results
    
    Args:
        history: training history dictionary
        eval_data: evaluation results dictionary
        args: arguments
        save_dir: directory to save plots
    """
    print(f"\n📊 Creating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss breakdown
    axes[0, 1].plot(history['train_reconstruction_loss'], label='Train Reconstruction', alpha=0.8)
    axes[0, 1].plot(history['val_reconstruction_loss'], label='Val Reconstruction', alpha=0.8)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L2 regularization loss
    axes[0, 2].plot(history['train_l2_loss'], alpha=0.8, color='red')
    axes[0, 2].set_title('L2 Regularization Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('L2 Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Cosine similarity
    axes[1, 0].plot(history['val_cosine_similarity'], alpha=0.8, color='green')
    axes[1, 0].set_title('Validation Cosine Similarity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].semilogy(history['learning_rate'], alpha=0.8, color='purple')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Reconstruction MSE distribution
    axes[1, 2].hist(eval_data['reconstruction_mses'], bins=50, alpha=0.7, color='orange')
    axes[1, 2].set_title('Reconstruction MSE Distribution')
    axes[1, 2].set_xlabel('MSE')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature reconstruction comparison
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    feature_names = [
        'prev_acc_x', 'prev_acc_y', 'prev_acc_z',
        'prev_gyro_x', 'prev_gyro_y', 'prev_gyro_z',
        'curr_acc_x', 'curr_acc_y', 'curr_acc_z',
        'curr_gyro_x', 'curr_gyro_y', 'curr_gyro_z'
    ]
    
    # Sample a subset for visualization
    n_samples = min(1000, len(eval_data['originals']))
    sample_indices = np.random.choice(len(eval_data['originals']), n_samples, replace=False)
    
    for i, name in enumerate(feature_names):
        original_feature = eval_data['originals'][sample_indices, i]
        reconstructed_feature = eval_data['reconstructed'][sample_indices, i]
        
        axes[i].scatter(original_feature, reconstructed_feature, alpha=0.5, s=1)
        axes[i].plot([original_feature.min(), original_feature.max()], 
                    [original_feature.min(), original_feature.max()], 
                    'r--', alpha=0.8)
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Original')
        axes[i].set_ylabel('Reconstructed')
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(original_feature, reconstructed_feature)[0, 1]
        axes[i].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Latent space visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sample for visualization
    n_viz_samples = min(2000, len(eval_data['latent']))
    viz_indices = np.random.choice(len(eval_data['latent']), n_viz_samples, replace=False)
    latent_sample = eval_data['latent'][viz_indices]
    
    # PCA of latent space
    if args.latent_dim > 2:
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_sample)
        explained_variance = pca.explained_variance_ratio_
        
        axes[0].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6, s=1)
        axes[0].set_title(f'Latent Space PCA\n(Explained variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f})')
        axes[0].set_xlabel(f'PC1 ({explained_variance[0]:.3f})')
        axes[0].set_ylabel(f'PC2 ({explained_variance[1]:.3f})')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].scatter(latent_sample[:, 0], latent_sample[:, 1] if latent_sample.shape[1] > 1 else np.zeros_like(latent_sample[:, 0]), alpha=0.6, s=1)
        axes[0].set_title('Latent Space (2D)')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].grid(True, alpha=0.3)
    
    # t-SNE of latent space (if computationally feasible)
    if args.latent_dim > 2 and n_viz_samples <= 1000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_viz_samples//4))
        latent_tsne = tsne.fit_transform(latent_sample)
        
        axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6, s=1)
        axes[1].set_title('Latent Space t-SNE')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 't-SNE skipped\n(too many samples\nor latent_dim ≤ 2)', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('t-SNE Not Computed')
    
    # Latent dimension statistics
    latent_means = np.mean(latent_sample, axis=0)
    latent_stds = np.std(latent_sample, axis=0)
    
    x_pos = np.arange(len(latent_means))
    axes[2].bar(x_pos, latent_means, yerr=latent_stds, alpha=0.7, capsize=5)
    axes[2].set_title('Latent Dimension Statistics')
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('Mean ± Std')
    axes[2].set_xticks(x_pos)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_space.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {save_dir}/")

def save_model_and_results(model, dataset, history, eval_data, args, save_dir="imu_autoencoder_results"):
    """Save trained model and results"""
    print(f"\n💾 Saving model and results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = f"{save_dir}/imu_autoencoder_latent{args.latent_dim}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_args': {
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'l2_reg': model.l2_reg
        },
        'training_args': vars(args),
        'final_metrics': {
            'reconstruction_mse': float(eval_data['overall_mse']),
            'cosine_similarity': float(eval_data['overall_cosine_sim'])
        }
    }, model_path)
    
    # Save scaler
    scaler_path = f"{save_dir}/imu_scaler.pkl"
    import joblib
    joblib.dump(dataset.scaler, scaler_path)
    
    # Save training history (convert numpy types to native Python types)
    history_path = f"{save_dir}/training_history.json"
    history_converted = convert_numpy_types(history)
    with open(history_path, 'w') as f:
        json.dump(history_converted, f, indent=2)
    
    # Save evaluation metrics
    metrics_path = f"{save_dir}/evaluation_metrics.json"
    metrics = {
        'reconstruction_mse': float(eval_data['overall_mse']),
        'cosine_similarity': float(eval_data['overall_cosine_sim']),
        'mse_std': float(np.std(eval_data['reconstruction_mses'])),
        'cosine_sim_std': float(np.std(eval_data['cosine_similarities'])),
        'latent_dim': args.latent_dim,
        'training_samples': len(dataset),
        'timestamp': datetime.now().isoformat()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ History saved to: {history_path}")
    print(f"✅ Metrics saved to: {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='IMU Autoencoder Training')
    parser.add_argument('--data_dir', type=str, default='../data_collection',
                       help='Path to data collection directory')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension size (default: 32)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--l2_reg', type=float, default=1e-4,
                       help='L2 regularization on latent (default: 1e-4)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum trajectories to load (None for all)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda) (default: auto)')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🌙 IMU Autoencoder for Lunar Rover Data")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    try:
        # Load IMU data
        imu_data = load_imu_data_from_missions(args.data_dir, args.max_trajectories)
        analyze_imu_data(imu_data)
        
        # Create dataset
        dataset = IMUDataset(imu_data)
        
        # Split data
        n_total = len(dataset)
        n_train = int(args.train_split * n_total)
        n_val = int(args.val_split * n_total)
        n_test = n_total - n_train - n_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        print(f"\n📊 Data splits:")
        print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/n_total*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/n_total*100:.1f}%)")
        print(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/n_total*100:.1f}%)")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        model = IMUAutoencoder(input_dim=12, latent_dim=args.latent_dim, l2_reg=args.l2_reg)
        model.to(device)
        
        print(f"\n🏗️  Model architecture:")
        print(f"  Input dimension: {model.input_dim}")
        print(f"  Latent dimension: {model.latent_dim}")
        print(f"  L2 regularization: {model.l2_reg}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history = train_autoencoder(model, train_loader, val_loader, device, args)
        
        # Evaluate model
        eval_data = evaluate_autoencoder(model, test_loader, dataset, device, args)
        
        # Create visualizations
        visualize_results(history, eval_data, args)
        
        # Save model and results
        save_model_and_results(model, dataset, history, eval_data, args)
        
        print("\n🎉 Training and evaluation completed successfully!")
        print(f"Final metrics:")
        print(f"  Reconstruction MSE: {eval_data['overall_mse']:.6f}")
        print(f"  Cosine Similarity: {eval_data['overall_cosine_sim']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 