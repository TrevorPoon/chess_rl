import os
import random
from collections import deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- Utility Functions ----------------

def board_to_feature_vector(board):
    """
    Convert a chess board into a 768-dimensional feature vector.
    
    The 768 features are organized as follows:
      - The first 384 features represent white pieces.
      - The next 384 features represent black pieces.
    
    Each side has 6 piece types (pawn, knight, bishop, rook, queen, king)
    and 64 squares. For a given piece on a square, the corresponding feature
    is set to 1, and all other features remain 0.
    """
    feature = np.zeros(768, dtype=np.float32)
    # Piece types: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
    # For white, use offset 0; for black, use offset 384.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type  # 1 to 6
            offset = 0 if piece.color == chess.WHITE else 384
            # Each piece type occupies 64 slots.
            index = offset + (piece_type - 1) * 64 + square
            feature[index] = 1.0
    return feature

def get_feature_vector_size():
    return 768

def get_move_space_size():
    # For NNUE evaluation we do not use a policy head,
    # so we return a dummy value.
    return 1

# ---------------- NNUE Model Definition ----------------

class NNUEModel(nn.Module):
    """NNUE-style neural network model for board evaluation."""
    
    def __init__(self):
        super().__init__()
        input_size = get_feature_vector_size()  # 768 features
        # NNUE typically uses a single hidden layer.
        self.fc1 = nn.Linear(input_size, 256)
        # The output is a single scalar evaluation.
        self.out = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.out(x)
        return value

# ---------------- Replay Buffer ----------------

class ReplayBuffer:
    """A simple replay buffer to store training examples."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, value):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self.buffer.append((state, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, values = zip(*batch)
        return (torch.stack(states).to('cpu'),
                torch.stack(values).to('cpu'))
    
    def __len__(self):
        return len(self.buffer)

# ---------------- NNUE Chess Agent ----------------

class NNUEChessAgent:
    """Chess agent that uses a NNUE-style network for board evaluation."""
    
    def __init__(self,
                 gamma=0.99,
                 l2_lambda=1e-4,
                 replay_capacity=10000,
                 optimizer_cls=optim.Adam):
        """
        Initialize key variables.
        
        Args:
            gamma (float): Discount factor (if used for bootstrapping in training).
            l2_lambda (float): L2 regularization strength.
            replay_capacity (int): Capacity of the replay buffer.
            optimizer_cls (torch.optim.Optimizer): Optimizer class.
        """
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNUEModel().to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters())
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def board_to_vector(self, board):
        """Convert a chess board into a 768-dimensional feature vector."""
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

    def select_move(self, board, temperature=None, noise_weight=None):
        """
        Select a move by evaluating each legal move and converting evaluations into move 
        probabilities using temperature scaling and optional Dirichlet noise.
        
        Args:
            board (chess.Board): Current board state.
            temperature (float, optional): Temperature for scaling probabilities.
                Defaults to 1.0.
            noise_weight (float, optional): Weight for Dirichlet noise.
                Defaults to 0.0.
        
        Returns:
            tuple: (chosen_move, move_probability_distribution)
        """
        # Set default parameters if not provided.
        temperature = temperature if temperature is not None else 1.0
        noise_weight = noise_weight if noise_weight is not None else 0.0

        legal_moves = list(board.legal_moves)
        evals = []
        
        # Evaluate each legal move.
        for move in legal_moves:
            board.push(move)
            state = self.board_to_vector(board)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                eval_value = self.model(state_tensor).item()
            evals.append(eval_value)
            board.pop()

        # Convert evaluations to a tensor.
        evals_tensor = torch.tensor(evals, dtype=torch.float32, device=self.device)

        # Handle the special case when temperature is zero (greedy selection).
        if temperature == 0:
            if board.turn:  # White to move: choose highest evaluation.
                best_index = torch.argmax(evals_tensor)
            else:           # Black to move: choose lowest evaluation.
                best_index = torch.argmin(evals_tensor)
            move_probs = torch.zeros(len(legal_moves), device=self.device)
            move_probs[best_index] = 1.0
            chosen_index = best_index.item()
            return legal_moves[chosen_index], move_probs.cpu().numpy()
        
        # Adjust evaluations based on the side to move.
        # For white, higher evaluations are better; for black, lower evaluations are better.
        if board.turn:  # white to move
            scaled = evals_tensor / temperature
        else:           # black to move: invert evaluation values so that lower becomes better
            scaled = -evals_tensor / temperature

        # Convert scaled evaluations into a probability distribution using softmax.
        move_probs = torch.softmax(scaled, dim=0)

        # Optionally add Dirichlet noise.
        if noise_weight > 0:
            noise = torch.tensor(
                np.random.dirichlet([0.03] * len(legal_moves)),
                dtype=torch.float32, device=self.device
            )
            move_probs = (1 - noise_weight) * move_probs + noise_weight * noise
            move_probs = move_probs / (move_probs.sum() + 1e-8)

        # Sample a move according to the computed probabilities.
        chosen_index = torch.multinomial(move_probs, 1).item()
        return legal_moves[chosen_index], move_probs.cpu().numpy()



    def train(self):
        """Set the model to training mode."""
        self.model.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def move_to_index(self, move):
        # Since NNUE doesn't use a policy head, always return 0.
        return 0

    def minimise_loss(self, states, _, values, batch_size=32):
        """
        Compute the loss as the mean squared error (MSE) between target and predicted evaluations,
        plus L2 regularization.
        """
        pred_values = self.model(states).squeeze()
        loss = torch.mean((values - pred_values) ** 2)
        l2_reg = self.l2_lambda * sum(torch.norm(param) ** 2 for param in self.model.parameters())
        total_loss = loss + l2_reg
        return total_loss

    def train_step(self, batch_size=32):
        """
        Perform a single training step using a batch from the replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return

        self.optimizer.zero_grad()
        states, values = self.replay_buffer.sample(batch_size)
        total_loss = self.minimise_loss(states, values, batch_size)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def save_model(self, model_name="nnue_chess_model.pth"):
        """
        Save the model state to a file in a 'model' directory.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to '{save_path}'")


