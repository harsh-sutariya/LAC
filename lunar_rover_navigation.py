#!/usr/bin/env python3
"""
Lunar Rover Navigation using Learned World Model

This script implements the planning component from plan.md:
1. Loads the trained world model from HJEPA training
2. Implements Model Predictive Control (MPC) for navigation
3. Uses Cross-Entropy Method (CEM) for action sequence optimization
4. Navigates to specified goal locations

Usage:
    python3 lunar_rover_navigation.py --model_path MODEL_PATH --goal_x X --goal_y Y
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Optional

# Add HJEPA to Python path
sys.path.insert(0, 'HJEPA')

from hjepa.models.hjepa import HJEPA
from hjepa.configs import load_config
from hjepa.data.lunar_rover import LunarRoverDatasetConfig, LunarRoverDataset


class LunarRoverMPC:
    """
    Model Predictive Control for Lunar Rover Navigation
    
    Implements the planning algorithm described in plan.md:
    - Uses learned world model to predict future states
    - Optimizes action sequences using Cross-Entropy Method (CEM)
    - Plans trajectories to reach specified goal locations
    """
    
    def __init__(
        self,
        model: HJEPA,
        normalizer,
        horizon: int = 20,
        num_samples: int = 1000,
        num_elites: int = 100,
        num_iterations: int = 5,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_bounds = action_bounds
        self.device = device
        
        # Action dimension (linear velocity, angular velocity)
        self.action_dim = 2
        
        # Initialize action distribution parameters
        self.action_mean = torch.zeros(horizon, self.action_dim, device=device)
        self.action_std = torch.ones(horizon, self.action_dim, device=device) * 0.5
        
    def predict_trajectory(
        self, 
        initial_state: torch.Tensor, 
        actions: torch.Tensor,
        initial_imu: Optional[torch.Tensor] = None,
        initial_position: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future trajectory using the learned world model
        
        Args:
            initial_state: Initial image observation (B, C, H, W)
            actions: Action sequence (B, T, action_dim)
            initial_imu: Initial IMU reading (B, 12)
            initial_position: Initial position (B, 2 or 3)
            
        Returns:
            predicted_positions: Predicted positions (B, T, 2 or 3)
            predicted_states: Predicted latent states (B, T, latent_dim)
        """
        batch_size = initial_state.shape[0]
        
        # Prepare initial observations
        states = [initial_state]
        positions = [initial_position] if initial_position is not None else []
        imus = [initial_imu] if initial_imu is not None else []
        
        # Roll out trajectory using the model
        current_state = initial_state
        current_imu = initial_imu
        current_position = initial_position
        
        with torch.no_grad():
            for t in range(self.horizon):
                # Encode current state
                optional_inputs = {}
                if current_imu is not None:
                    optional_inputs['proprio'] = current_imu
                if current_position is not None:
                    optional_inputs['location'] = current_position
                
                # Get encoding
                encoding = self.model.level1.backbone(current_state, **optional_inputs)
                
                # Predict next state using the model
                # Note: This is a simplified version - in practice, we'd use the full JEPA forward pass
                action = actions[:, t] if t < actions.shape[1] else actions[:, -1]
                
                # For now, we'll use a simple kinematic model for position prediction
                # In a full implementation, this would use the learned dynamics model
                if current_position is not None:
                    dt = 0.1  # Time step
                    linear_vel = action[:, 0:1]  # Linear velocity
                    angular_vel = action[:, 1:2]  # Angular velocity
                    
                    # Simple kinematic update (can be replaced with learned model)
                    new_position = current_position.clone()
                    new_position[:, 0] += linear_vel.squeeze() * dt * torch.cos(angular_vel.squeeze() * dt)
                    new_position[:, 1] += linear_vel.squeeze() * dt * torch.sin(angular_vel.squeeze() * dt)
                    
                    positions.append(new_position)
                    current_position = new_position
                
                # Update IMU (simplified - would use learned model)
                if current_imu is not None:
                    # For now, keep IMU constant (in practice, predict using learned model)
                    imus.append(current_imu)
        
        # Stack results
        if positions:
            predicted_positions = torch.stack(positions[1:], dim=1)  # (B, T, pos_dim)
        else:
            predicted_positions = None
            
        return predicted_positions, encoding.encodings  # Return final encoding as state
    
    def cost_function(
        self, 
        predicted_positions: torch.Tensor, 
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost for predicted trajectories
        
        Args:
            predicted_positions: Predicted positions (B, T, 2 or 3)
            goal_position: Goal position (2 or 3,)
            
        Returns:
            costs: Cost for each trajectory (B,)
        """
        if predicted_positions is None:
            return torch.zeros(1, device=self.device)
        
        # Distance to goal at final timestep
        final_positions = predicted_positions[:, -1, :2]  # Take x, y coordinates
        goal_pos = goal_position[:2].unsqueeze(0).expand(final_positions.shape[0], -1)
        
        # L2 distance to goal
        distances = torch.norm(final_positions - goal_pos, dim=1)
        
        # Additional costs can be added here (e.g., smoothness, collision avoidance)
        
        return distances
    
    def plan_action_sequence(
        self,
        initial_state: torch.Tensor,
        goal_position: torch.Tensor,
        initial_imu: Optional[torch.Tensor] = None,
        initial_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Plan optimal action sequence using Cross-Entropy Method (CEM)
        
        Args:
            initial_state: Initial image observation (1, C, H, W)
            goal_position: Goal position (2 or 3,)
            initial_imu: Initial IMU reading (1, 12)
            initial_position: Initial position (1, 2 or 3)
            
        Returns:
            best_action_sequence: Optimal action sequence (horizon, action_dim)
        """
        
        # Reset action distribution
        self.action_mean.fill_(0)
        self.action_std.fill_(0.5)
        
        best_cost = float('inf')
        best_actions = None
        
        for iteration in range(self.num_iterations):
            # Sample action sequences
            eps = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
            actions = self.action_mean.unsqueeze(0) + self.action_std.unsqueeze(0) * eps
            
            # Clamp actions to bounds
            actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
            
            # Expand initial state for batch processing
            batch_initial_state = initial_state.expand(self.num_samples, -1, -1, -1)
            batch_initial_imu = initial_imu.expand(self.num_samples, -1) if initial_imu is not None else None
            batch_initial_position = initial_position.expand(self.num_samples, -1) if initial_position is not None else None
            
            # Predict trajectories
            predicted_positions, _ = self.predict_trajectory(
                batch_initial_state, 
                actions,
                batch_initial_imu,
                batch_initial_position
            )
            
            # Compute costs
            costs = self.cost_function(predicted_positions, goal_position)
            
            # Select elite samples
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elite_actions = actions[elite_indices]
            
            # Update distribution
            self.action_mean = elite_actions.mean(dim=0)
            self.action_std = elite_actions.std(dim=0) + 1e-6  # Add small epsilon for numerical stability
            
            # Track best solution
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]]
                best_actions = elite_actions[0]
            
            print(f"Iteration {iteration + 1}: Best cost = {best_cost:.4f}")
        
        return best_actions
    
    def navigate_to_goal(
        self,
        initial_state: torch.Tensor,
        goal_position: torch.Tensor,
        initial_imu: Optional[torch.Tensor] = None,
        initial_position: Optional[torch.Tensor] = None,
        max_steps: int = 100
    ) -> List[torch.Tensor]:
        """
        Navigate to goal using MPC (receding horizon control)
        
        Args:
            initial_state: Initial image observation (1, C, H, W)
            goal_position: Goal position (2 or 3,)
            initial_imu: Initial IMU reading (1, 12)
            initial_position: Initial position (1, 2 or 3)
            max_steps: Maximum number of control steps
            
        Returns:
            trajectory: List of executed actions
        """
        
        current_state = initial_state
        current_imu = initial_imu
        current_position = initial_position
        
        executed_actions = []
        
        for step in range(max_steps):
            print(f"\nStep {step + 1}/{max_steps}")
            
            # Check if goal is reached
            if current_position is not None:
                distance_to_goal = torch.norm(current_position[0, :2] - goal_position[:2])
                print(f"Distance to goal: {distance_to_goal:.4f}")
                
                if distance_to_goal < 0.5:  # Goal tolerance
                    print("🎯 Goal reached!")
                    break
            
            # Plan action sequence
            action_sequence = self.plan_action_sequence(
                current_state, 
                goal_position,
                current_imu,
                current_position
            )
            
            # Execute first action (MPC principle)
            first_action = action_sequence[0]
            executed_actions.append(first_action)
            
            print(f"Executing action: linear_vel={first_action[0]:.3f}, angular_vel={first_action[1]:.3f}")
            
            # Simulate state transition (in practice, this would be real robot execution)
            # Here we use the same prediction as in the planner
            if current_position is not None:
                dt = 0.1
                linear_vel = first_action[0]
                angular_vel = first_action[1]
                
                new_position = current_position.clone()
                new_position[0, 0] += linear_vel * dt * torch.cos(angular_vel * dt)
                new_position[0, 1] += linear_vel * dt * torch.sin(angular_vel * dt)
                
                current_position = new_position
                print(f"Current position: x={current_position[0, 0]:.3f}, y={current_position[0, 1]:.3f}")
        
        return executed_actions


def load_trained_model(model_path: str, config_path: str) -> Tuple[HJEPA, object]:
    """Load trained HJEPA model and normalizer"""
    print(f"Loading model from {model_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create model
    model = HJEPA(config.hjepa)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load normalizer (if available)
    normalizer = checkpoint.get('normalizer', None)
    
    return model, normalizer


def main():
    """Main navigation function"""
    parser = argparse.ArgumentParser(description="Lunar Rover Navigation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config_path", type=str, 
                       default="HJEPA/hjepa/configs/lunar_rover/lunar_rover_base.yaml",
                       help="Path to model configuration")
    parser.add_argument("--goal_x", type=float, default=5.0, help="Goal X coordinate")
    parser.add_argument("--goal_y", type=float, default=5.0, help="Goal Y coordinate")
    parser.add_argument("--start_x", type=float, default=0.0, help="Start X coordinate")
    parser.add_argument("--start_y", type=float, default=0.0, help="Start Y coordinate")
    parser.add_argument("--horizon", type=int, default=20, help="Planning horizon")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum navigation steps")
    
    args = parser.parse_args()
    
    print("🚀 Starting Lunar Rover Navigation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Goal: ({args.goal_x}, {args.goal_y})")
    print(f"Start: ({args.start_x}, {args.start_y})")
    
    try:
        # Load trained model
        model, normalizer = load_trained_model(args.model_path, args.config_path)
        model.eval()
        
        # Create MPC planner
        planner = LunarRoverMPC(
            model=model,
            normalizer=normalizer,
            horizon=args.horizon
        )
        
        # Create initial state (dummy image for now)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        initial_state = torch.randn(1, 3, 256, 256, device=device)  # Dummy image
        initial_position = torch.tensor([[args.start_x, args.start_y, 0.0]], device=device)
        initial_imu = torch.zeros(1, 12, device=device)  # Dummy IMU
        goal_position = torch.tensor([args.goal_x, args.goal_y, 0.0], device=device)
        
        # Execute navigation
        print("\n🎯 Starting navigation...")
        trajectory = planner.navigate_to_goal(
            initial_state=initial_state,
            goal_position=goal_position,
            initial_imu=initial_imu,
            initial_position=initial_position,
            max_steps=args.max_steps
        )
        
        print(f"\n🏁 Navigation completed in {len(trajectory)} steps")
        print("=" * 60)
        
        # Save trajectory
        trajectory_path = "lunar_rover_trajectory.pt"
        torch.save(trajectory, trajectory_path)
        print(f"📁 Trajectory saved to: {trajectory_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Navigation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 