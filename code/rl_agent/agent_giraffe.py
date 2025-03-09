import os
import random
from collections import deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Utility functions (to be implemented in utils/util.py)
from utils.util import board_to_feature_vector, get_feature_vector_size, get_move_space_size

# Set the number of torch threads
torch.set_num_threads(os.cpu_count())


class GiraffeNet(nn.Module):
    """Neural network with two heads: policy (move probabilities) and value (board evaluation)."""
    
    def __init__(self):
        super().__init__()
        input_size = get_feature_vector_size()  # e.g., 363 features
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, get_move_space_size())
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            tuple: (policy, value) where policy is softmax over moves and value is a tanh activation.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value


class ReplayBuffer:
    """A simple replay buffer to store training examples."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, policy, value):
        """Push a new sample into the buffer."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(policy, torch.Tensor):
            policy = torch.tensor(policy, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        """Sample a random batch from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (torch.stack(states).to('cpu'),
                torch.stack(policies).to('cpu'),
                torch.stack(values).to('cpu'))
    
    def __len__(self):
        return len(self.buffer)


class GiraffeChessAgent:
    """Chess agent that uses a neural network for move prediction and board evaluation."""
    
    def __init__(self,
                 gamma=0.99,
                 l2_lambda=1e-4,
                 default_temperature=1.0,
                 default_noise_weight=0.25,
                 replay_capacity=10000,
                 optimizer_cls=optim.Adam,
                 lr=1e-4):
        """
        Initialize all key variables in one place.
        
        Args:
            gamma (float): Discount factor for bootstrapping rewards.
            l2_lambda (float): L2 regularization strength.
            default_temperature (float): Default temperature for move selection.
            default_noise_weight (float): Default weight for Dirichlet noise.
            replay_capacity (int): Capacity of the replay buffer.
            optimizer_cls (torch.optim.Optimizer): Optimizer class to use.
        """
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.default_temperature = default_temperature
        self.default_noise_weight = default_noise_weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GiraffeNet().to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=self.l2_lambda)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',       # we want to reduce LR when loss stops decreasing
            factor=0.99,       # reduce LR by a factor of 0.5
            patience=10,     # wait for 100 steps before reducing LR
            verbose=False
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def board_to_vector(self, board):
        """Convert a chess board into a feature vector."""
        return board_to_feature_vector(board)

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
        """
        Convert a chess move to an index in the policy vector.
        
        Regular moves: first 4096 indices (64*64)
        Promotion moves: next 2048 indices (4 pieces * 64 * 8)
        """
        src = move.from_square
        dst = move.to_square
        promotion = move.promotion

        # Base index for a regular move
        index = src * 64 + dst

        # Handle promotions
        if promotion:
            promotion_base = 64 * 64
            move_index = src * 8 + dst % 8
            if promotion == chess.QUEEN:
                index = promotion_base + move_index
            elif promotion == chess.ROOK:
                index = promotion_base + (64 * 8) + move_index
            elif promotion == chess.BISHOP:
                index = promotion_base + (2 * 64 * 8) + move_index
            elif promotion == chess.KNIGHT:
                index = promotion_base + (3 * 64 * 8) + move_index

        # Safety check
        if index >= get_move_space_size():
            print(f"Warning: move {move} produced invalid index {index}")
            index = 0
            
        return index

    def train(self):
        """Set the model to training mode."""
        self.model.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def select_move(self, board, temperature=None, noise_weight=None):
        """
        Select a move using the current policy with temperature scaling and Dirichlet noise.
        
        Args:
            board (chess.Board): Current board state.
            temperature (float, optional): Temperature for scaling probabilities.
                Defaults to self.default_temperature.
            noise_weight (float, optional): Weight for Dirichlet noise.
                Defaults to self.default_noise_weight.
            
        Returns:
            tuple: Selected move and the corresponding move probability distribution.
        """
        temperature = temperature if temperature is not None else self.default_temperature
        noise_weight = noise_weight if noise_weight is not None else self.default_noise_weight
        
        # Convert board state to tensor
        state = self.board_to_vector(board)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get policy from the model
        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        legal_moves = list(board.legal_moves)
        move_indices = [self.move_to_index(move) for move in legal_moves]
        move_indices = [index if index < policy.size(1) else 0 for index in move_indices]
        move_indices_tensor = torch.tensor(move_indices, dtype=torch.long, device=self.device)

        move_probs = policy[0].index_select(0, move_indices_tensor)
        move_probs = torch.clamp(move_probs, min=0)
        move_probs = move_probs ** (1.0 / (temperature + 1e-10))

        if noise_weight > 0:
            noise = torch.tensor(
                np.random.dirichlet([0.03] * len(legal_moves)),
                dtype=torch.float32, device=self.device
            )
            move_probs = (1 - noise_weight) * move_probs + noise_weight * noise

        move_probs = move_probs / (move_probs.sum() + 1e-8)
        chosen_index = torch.multinomial(move_probs, 1).item()
        return legal_moves[chosen_index], move_probs.cpu().numpy()

    def _pad_policy(self, pred_policies, policies):
        """
        Pad the policies tensor if the predicted policy has more dimensions.
        
        Args:
            pred_policies (Tensor): Predicted policy from the model.
            policies (Tensor): Target policies.
            
        Returns:
            Tensor: Padded target policies.
        """
        if pred_policies.size(1) > policies.size(1):
            padding = torch.zeros(
                policies.size(0), pred_policies.size(1) - policies.size(1),
                device=self.device
            )
            policies = torch.cat([policies, padding], dim=1)
        return policies

    def minimise_loss(self, states, policies, values, batch_size=32):
        """
        Compute the total loss (policy + value + L2 regularization).
        
        Args:
            states (Tensor): Batch of state vectors.
            policies (Tensor): Batch of target policies.
            values (Tensor): Batch of target values.
            batch_size (int): Batch size.
            
        Returns:
            Tensor: Total loss.
        """
        pred_policies, pred_values = self.model(states)
        policies = self._pad_policy(pred_policies, policies)
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
        value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
        l2_reg = self.l2_lambda * sum(torch.norm(param) ** 2 for param in self.model.parameters())
        total_loss = policy_loss + value_loss + l2_reg
        
        return total_loss

    def train_step(self, batch_size=32):
        """
        Perform a single training step using a batch of samples from the replay buffer.
        
        Args:
            batch_size (int): Number of samples in the batch.
            
        Returns:
            float: The total loss value (if training is performed).
        """
        if len(self.replay_buffer) < batch_size:
            return

        self.optimizer.zero_grad()
        states, policies, values = self.replay_buffer.sample(batch_size)
        total_loss = self.minimise_loss(states, policies, values, batch_size)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def save_model(self, model_name="giraffe_chess_model.pth"):
        """
        Save the model state to a file in the 'model' directory.
        
        Args:
            model_name (str): Name of the saved model file.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = os.path.join(base_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to '{save_path}'")
