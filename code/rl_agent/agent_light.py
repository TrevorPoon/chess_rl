import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import numpy as np
from collections import deque
import random
import os

from utils.util import board_to_tensor, get_move_space_size

class ChessNet(nn.Module):
    def __init__(self, use_batch_norm=True):
        super(ChessNet, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.policy_head = nn.Linear(512, get_move_space_size())
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = torch.relu(x)
        
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        # Move tensors to the specified device
        return (torch.stack(states).to(device), 
                torch.stack(policies).to(device),
                torch.stack(values).to(device))
    
    def __len__(self):
        return len(self.buffer)

class ChessLightAgent:
    def __init__(self, device=None):
        # Allow device flexibility: use GPU if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = ChessNet(use_batch_norm=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Optional learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.replay_buffer = ReplayBuffer()
        # Cache for move-to-index mapping to reduce recomputation
        self.move_to_index_cache = {}

    def load_model(self, model_path):
        """Load the model parameters from a checkpoint file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise e

    def save_model(self, model_name="chess_model.pth"):
        """Save the model to the 'model' directory in the parent directory"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))
        print(f"Model saved to '{model_dir}/{model_name}'")

    def move_to_index(self, move):
        """Convert a chess move to an index in the policy vector.
        Regular moves: first 4096 indices (64*64)
        Promotion moves: next 2048 indices (4 pieces * 64 * 8)
        Uses caching to avoid recomputation.
        """
        key = str(move)
        if key in self.move_to_index_cache:
            return self.move_to_index_cache[key]

        src = move.from_square
        dst = move.to_square
        promotion = move.promotion
        
        # Base index for the move
        index = src * 64 + dst
        
        # Handle promotions
        if promotion:
            promotion_base = 64 * 64
            if promotion == chess.QUEEN:
                index = promotion_base + (src * 8 + dst % 8)
            elif promotion == chess.ROOK:
                index = promotion_base + (64 * 8) + (src * 8 + dst % 8)
            elif promotion == chess.BISHOP:
                index = promotion_base + (2 * 64 * 8) + (src * 8 + dst % 8)
            elif promotion == chess.KNIGHT:
                index = promotion_base + (3 * 64 * 8) + (src * 8 + dst % 8)
                
        # Safety check
        if index >= get_move_space_size():
            print(f"Warning: move {move} produced invalid index {index}")
            index = 0
            
        self.move_to_index_cache[key] = index
        return index

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        print("Model set to evaluation mode.")

    def select_move(self, board, temperature=1.0):
        """Select a move using the current policy"""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)
        
        legal_moves = list(board.legal_moves)
        move_probs = torch.zeros(len(legal_moves), device=self.device)
        
        for i, move in enumerate(legal_moves):
            move_idx = self.move_to_index(move)
            if move_idx < policy[0].size(0):
                move_probs[i] = policy[0][move_idx]
        
        # Apply temperature scaling
        move_probs = move_probs ** (1 / (temperature + 1e-10))
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
        else:
            move_probs = torch.ones_like(move_probs) / len(move_probs)
        
        selected_idx = torch.multinomial(move_probs, 1).item()
        return legal_moves[selected_idx]
    
    def train_step(self, batch_size=32):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, policies, values = self.replay_buffer.sample(batch_size, self.device)
        
        # Forward pass
        pred_policies, pred_values = self.model(states)
        
        # Ensure target policies and predicted policies have the same dimensions
        if pred_policies.size(1) > policies.size(1):
            padding = torch.zeros(policies.size(0), 
                                  pred_policies.size(1) - policies.size(1),
                                  device=self.device)
            policies = torch.cat([policies, padding], dim=1)
        
        # Calculate losses:
        # Policy loss: cross-entropy style loss (using the negative log likelihood)
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
        # Value loss: Mean Squared Error
        value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
        total_loss = policy_loss + value_loss
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()  # update learning rate
        
        return total_loss.item()
