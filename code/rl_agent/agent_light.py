import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import os

from utils.util import board_to_tensor, get_move_space_size

# Optimize CPU utilization: use all available CPU cores.
torch.set_num_threads(os.cpu_count())

# --- Residual Block Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = torch.relu(out)
        return out

# --- Enhanced ChessNet with Residual Blocks ---
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(64, 64, use_dropout=True, dropout_prob=0.3)
        self.res2 = ResidualBlock(64, 128, use_dropout=True, dropout_prob=0.3)
        self.res3 = ResidualBlock(128, 128, use_dropout=True, dropout_prob=0.3)
        self.res4 = ResidualBlock(128, 256, use_dropout=True, dropout_prob=0.3)
        self.res5 = ResidualBlock(256, 256, use_dropout=True, dropout_prob=0.3)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout_fc = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.policy_head = nn.Linear(512, get_move_space_size())
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, policy, value):
        max_priority = max(self.priorities, default=1)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, policy, value))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, policy, value)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        states, policies, values = zip(*samples)
        states = torch.stack(states).to('cpu')
        policies = torch.stack(policies).to('cpu')
        values = torch.stack(values).to('cpu')
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        return indices, states, policies, values, weights

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# --- Chess Neural Agent ---
class ChessLightAgent:
    def __init__(self, policy_loss_weight=1.0, value_loss_weight=1.0):
        self.device = torch.device("cpu")
        self.model = ChessNet().to(self.device)
        # L2 regularization is added via weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-4)
        # Learning rate scheduler: adjust parameters as needed.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.replay_buffer = PrioritizedReplayBuffer()
        # Loss weighting coefficients
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight

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

        index = src * 64 + dst

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

        if index >= get_move_space_size():
            print(f"Warning: move {move} produced invalid index {index}")
            index = 0

        return index

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        print("Model set to evaluation mode.")

    def select_move(self, board, temperature=1.0):
        """Select a move using the current policy."""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)

        legal_moves = list(board.legal_moves)
        move_probs = torch.zeros(len(legal_moves))

        for i, move in enumerate(legal_moves):
            move_idx = self.move_to_index(move)
            if move_idx < policy[0].size(0):
                move_probs[i] = policy[0][move_idx]

        move_probs = move_probs ** (1/(temperature + 1e-10))
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
        else:
            move_probs = torch.ones_like(move_probs) / len(move_probs)

        move_idx = torch.multinomial(move_probs, 1).item()
        return legal_moves[move_idx]

    def train_step(self, batch_size=32, beta=0.4, accumulation_steps=1):
        """
        Perform one training step with prioritized replay, loss weighting, and gradient accumulation.
        The loss from each mini-batch is scaled by 1/accumulation_steps so that the overall gradient is comparable.
        """
        if len(self.replay_buffer) < batch_size * accumulation_steps:
            return None

        total_loss = 0.0
        # Accumulate gradients over several mini-batches
        for _ in range(accumulation_steps):
            indices, states, target_policies, target_values, weights = self.replay_buffer.sample(batch_size, beta=beta)

            pred_policies, pred_values = self.model(states)
            # If necessary, pad the target policies to match prediction size
            if pred_policies.size(1) > target_policies.size(1):
                padding = torch.zeros(target_policies.size(0), 
                                      pred_policies.size(1) - target_policies.size(1),
                                      device=self.device)
                target_policies = torch.cat([target_policies, padding], dim=1)

            # Compute per-sample losses
            log_preds = torch.log(pred_policies + 1e-8)
            per_sample_policy_loss = -torch.sum(target_policies * log_preds, dim=1)
            per_sample_value_loss = (target_values.squeeze() - pred_values.squeeze()) ** 2

            per_sample_loss = (self.policy_loss_weight * per_sample_policy_loss +
                               self.value_loss_weight * per_sample_value_loss)
            # Average the loss and scale it for accumulation
            loss = (per_sample_loss * weights).mean() / accumulation_steps

            # Backpropagate without updating parameters yet
            loss.backward()

            # Update replay buffer priorities for this mini-batch
            new_priorities = (per_sample_loss.detach().cpu().numpy() + 1e-6).tolist()
            self.replay_buffer.update_priorities(indices, new_priorities)

            # Accumulate the unscaled loss for logging purposes
            total_loss += loss.item() * accumulation_steps

        # Perform the optimizer step after accumulating gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update learning rate scheduler based on the total loss from accumulated steps
        self.scheduler.step(total_loss)

        return total_loss

    def save_model(self, model_name="chess_model.pth"):
        """Save the model to the 'model' directory in the parent directory."""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))
        print(f"Model saved to '{model_dir}/{model_name}'")
