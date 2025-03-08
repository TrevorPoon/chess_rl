import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import os

from utils.util import board_to_tensor, get_move_space_size

torch.set_num_threads(os.cpu_count())

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.policy_head = nn.Linear(512, get_move_space_size())
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
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
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (torch.stack(states).to('cpu'), 
                torch.stack(policies).to('cpu'),
                torch.stack(values).to('cpu'))
    
    def __len__(self):
        return len(self.buffer)


class ChessNeuralAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer()
    
    def board_to_vector(self, board):
        return board_to_tensor(board)

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

    def move_to_index(self, move):
        """Convert a chess move to an index in the policy vector.
        Regular moves: first 4096 indices (64*64)
        Promotion moves: next 2048 indices (4 pieces * 64 * 8)
        """
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
            
        return index
    
    def train(self):
        self.model.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def select_move(self, board, temperature=1.0, noise_weight=0.25):
        """Select a move using the current policy with optional Dirichlet noise for exploration."""
        state = self.board_to_vector(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)
        
        legal_moves = list(board.legal_moves)
        move_probs = torch.zeros(len(legal_moves), device=self.device)
        
        for i, move in enumerate(legal_moves):
            move_idx = self.move_to_index(move)
            if move_idx < policy[0].size(0):
                move_probs[i] = policy[0][move_idx]
        
        # Apply Dirichlet noise if needed
        if noise_weight > 0:
            noise = torch.tensor(np.random.dirichlet([0.03] * len(legal_moves)), 
                                dtype=torch.float32, device=self.device)
            move_probs = (1 - noise_weight) * move_probs + noise_weight * noise
        
        # Apply temperature scaling and renormalize
        move_probs = move_probs ** (1/(temperature + 1e-10))
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
        else:
            move_probs = torch.ones_like(move_probs) / len(move_probs)
        
        move_idx = torch.multinomial(move_probs, 1).item()
        return legal_moves[move_idx], move_probs

    
    def train_step(self, batch_size=32):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return
        
        self.optimizer.zero_grad()
        states, policies, values = self.replay_buffer.sample(batch_size)
        
        # Forward pass
        total_loss = self.minimise_loss(states, policies, values, batch_size)   
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def minimise_loss(self, states, policies, values, batch_size=32):
                # Use mixed precision if on GPU
        use_amp = self.device.type == 'cuda'
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                pred_policies, pred_values = self.model(states)
                if pred_policies.size(1) > policies.size(1):
                    padding = torch.zeros(policies.size(0), pred_policies.size(1) - policies.size(1), device=self.device)
                    policies = torch.cat([policies, padding], dim=1)
                policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
                value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
                total_loss = policy_loss + value_loss
            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            pred_policies, pred_values = self.model(states)
            if pred_policies.size(1) > policies.size(1):
                padding = torch.zeros(policies.size(0), pred_policies.size(1) - policies.size(1), device=self.device)
                policies = torch.cat([policies, padding], dim=1)
            policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
            value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
            total_loss = policy_loss + value_loss
        
        return total_loss
    
    def save_model(self, model_name="chess_model.pth"):
        """Save the model to the 'model' directory in the parent directory"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))
        print(f"Model saved to '{model_dir}/{model_name}'")