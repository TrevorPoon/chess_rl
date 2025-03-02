import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import os

# You must implement these functions in utils/util.py
# board_to_feature_vector(board) should convert a chess.Board to a flat vector (e.g. of length 363)
# get_feature_vector_size() should return the size of that vector
# get_move_space_size() should return the total number of indices in your move representation (e.g. 4096 + 2048)
from utils.util import board_to_feature_vector, get_feature_vector_size, get_move_space_size

torch.set_num_threads(os.cpu_count())

class GiraffeNet(nn.Module):
    def __init__(self):
        super(GiraffeNet, self).__init__()
        input_size = get_feature_vector_size()  # e.g., 363 features
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        # Two heads: one for policy (move probabilities) and one for value (board evaluation)
        self.policy_head = nn.Linear(256, get_move_space_size())
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, x):
        # x should be of shape (batch_size, input_size)
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
        # Convert each state from numpy array to torch tensor if necessary
        states = [torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state for state in states]
        return (torch.stack(states).to('cpu'),
                torch.stack(policies).to('cpu'),
                torch.stack(values).to('cpu'))
    
    def __len__(self):
        return len(self.buffer)

class GiraffeChessAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = GiraffeNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer()
    
    def board_to_vector(self, board):
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

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        print("Model set to evaluation mode.")

    def select_move(self, board, temperature=1.0, noise_weight=0.25):
        """
        Select a move using the current policy.
        The board is first converted to a feature vector using board_to_vector.
        A probability distribution over legal moves is constructed from the model's policy output.
        Temperature scaling and Dirichlet noise are applied to encourage exploration.
        """
        # Convert the board state to an input vector and then to a tensor.
        state = self.board_to_vector(board)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get policy (and value, if needed) from the model.
        with torch.no_grad():
            policy, value = self.model(state_tensor)

        # Extract legal moves and initialize probability tensor.
        legal_moves = list(board.legal_moves)
        move_probs = torch.zeros(len(legal_moves), dtype=torch.float32).to(self.device)

        # Map each legal move to its corresponding policy probability.
        for i, move in enumerate(legal_moves):
            move_idx = self.move_to_index(move)
            if move_idx < policy[0].size(0):
                move_probs[i] = policy[0][move_idx]
            else:
                # Assign a small probability if the index is out-of-range.
                move_probs[i] = 1e-8

        # Ensure all probabilities are non-negative.
        move_probs = torch.clamp(move_probs, min=0)

        # Apply temperature scaling.
        # We add a tiny epsilon to the temperature to avoid division by zero.
        temp = temperature + 1e-10
        move_probs = move_probs ** (1.0 / temp)

        # Normalize the probabilities, guarding against a zero sum.
        total = move_probs.sum()
        if total.item() == 0:
            move_probs = torch.ones_like(move_probs) / len(legal_moves)
        else:
            move_probs /= total

        # Add Dirichlet noise for exploration.
        if noise_weight > 0:
            noise = np.random.dirichlet([0.03] * len(legal_moves))
            noise_tensor = torch.tensor(noise, dtype=torch.float32).to(self.device)
            move_probs = (1 - noise_weight) * move_probs + noise_weight * noise_tensor
            
            # Re-normalize to ensure it sums to 1.
            total = move_probs.sum()
            if total.item() == 0:
                move_probs = torch.ones_like(move_probs) / len(legal_moves)
            else:
                move_probs /= total

        # Select a move based on the resulting probability distribution.
        chosen_index = torch.multinomial(move_probs, 1).item()
        return legal_moves[chosen_index]
    
    def train_step(self, batch_size=32):
        """Perform one training step using a mini-batch from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, policies, values = self.replay_buffer.sample(batch_size)
        
        pred_policies, pred_values = self.model(states)
        
        # Pad the target policy vector if needed
        if pred_policies.size(1) > policies.size(1):
            padding = torch.zeros(policies.size(0), pred_policies.size(1) - policies.size(1), device=self.device)
            policies = torch.cat([policies, padding], dim=1)
        
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
        value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def save_model(self, model_name="giraffe_chess_model.pth"):
        """
        Save the model state to a file in a 'model' directory in the parent directory.
        """
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to '{save_path}'")

if __name__ == "__main__":
    # Example usage:
    agent = GiraffeChessAgent()
    board = chess.Board()
    print("Initial board:")
    print(board)
    selected_move = agent.select_move(board)
    print("Selected move:", selected_move)
