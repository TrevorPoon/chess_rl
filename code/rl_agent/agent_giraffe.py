import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import os

# You must implement these functions in utils/util.py:
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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        # Convert policy to tensor if needed
        if not isinstance(policy, torch.Tensor):
            policy = torch.tensor(policy, dtype=torch.float32)
        # Ensure value is a tensor
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (torch.stack(states).to('cpu'),
                torch.stack(policies).to('cpu'),
                torch.stack(values).to('cpu'))
    
    def __len__(self):
        return len(self.buffer)

class GiraffeChessAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GiraffeNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99  # discount factor for bootstrapping rewards
        self.loss_penalty = 1.5  # extra penalty factor for moves that lose material
    
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
        # Precompute move indices for legal moves
        move_indices = [self.move_to_index(move) for move in legal_moves]
        move_indices = [index if index < policy[0].size(0) else 0 for index in move_indices]
        move_indices_tensor = torch.tensor(move_indices, dtype=torch.long, device=self.device)

        # Gather corresponding probabilities from the model's output
        move_probs = policy[0].index_select(0, move_indices_tensor)

        # Ensure all probabilities are non-negative.
        move_probs = torch.clamp(move_probs, min=0)

        # Temperature scaling
        move_probs = move_probs ** (1.0 / (temperature + 1e-10))

        # Apply Dirichlet noise if needed
        if noise_weight > 0:
            noise = torch.tensor(np.random.dirichlet([0.03] * len(legal_moves)), 
                                dtype=torch.float32, device=self.device)
            move_probs = (1 - noise_weight) * move_probs + noise_weight * noise

        # Normalize the probabilities once at the end
        move_probs = move_probs / (move_probs.sum() + 1e-8)

        # Select a move based on the resulting probability distribution.
        chosen_index = torch.multinomial(move_probs, 1).item()
        return legal_moves[chosen_index], move_probs.cpu().numpy()
    
    def self_play_episode(self):
        """
        Run a self-play episode without reward shaping from material score.
        Each move is assigned an immediate reward of 0, and only the terminal reward (win/loss/draw)
        is bootstrapped back over the moves.
        The episode transitions are stored in the replay buffer.
        """
        board = chess.Board()
        game_history = []
        
        while not board.is_game_over():
            # Save state before move
            state = self.board_to_vector(board)
            move, move_probs = self.select_move(board)
            board.push(move)
            
            # Immediate reward is set to 0 for all moves
            immediate_reward = 0.0
            
            game_history.append((state, move_probs, immediate_reward))
        
        # Terminal reward from game outcome
        result = board.result()
        if result == "1-0":
            terminal_reward = 1.0
        elif result == "0-1":
            terminal_reward = -1.0
        else:
            terminal_reward = 0.0

        # Bootstrapping: compute the return for each move (discounted sum of future rewards)
        G = terminal_reward
        returns = []
        # Reverse through the game history to compute cumulative returns
        for _, _, reward in reversed(game_history):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Store transitions in replay buffer. Here, the move_probs are used as the policy target,
        # and the return is used as the value target.
        for (state, move_probs, _), G in zip(game_history, returns):
            self.replay_buffer.push(state, move_probs, G)
        
        return board.result()  # Return game result for logging if desired


    def train_step(self, batch_size=32):
        
        if len(self.replay_buffer) < batch_size:
            return

        self.optimizer.zero_grad()
        states, policies, values = self.replay_buffer.sample(batch_size)
        
        return self.minimise_loss(states, policies, values, batch_size)

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

    def self_play_training(self, episodes=1000, train_steps_per_episode=10, batch_size=32):
        """
        Run self-play episodes and train the network.
        For every episode, self-play is used to generate game data with shaped rewards.
        After each episode, several training steps are performed.
        """
        for ep in range(1, episodes + 1):
            result = self.self_play_episode()
            # Run training steps using data collected in the episode
            losses = []
            for _ in range(train_steps_per_episode):
                loss = self.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)
            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode {ep}/{episodes} | Result: {result} | Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Example usage:
    agent = GiraffeChessAgent()
    
    # Optionally, run self-play training
    print("Starting self-play training...")
    agent.self_play_training(episodes=10, train_steps_per_episode=5, batch_size=32)
    
    # Example of loading a model and selecting a move
    board = chess.Board()
    print("Initial board:")
    print(board)
    selected_move, move_probs = agent.select_move(board)
    print("Selected move:", selected_move)
