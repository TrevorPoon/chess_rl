import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import os

from utils.util import board_to_tensor, get_move_space_size

##########################
# Neural Network Module  #
##########################
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Reduced network: two conv layers with fewer filters.
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Instead of three conv layers and two FC layers, we now use one FC layer.
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # Policy head outputs a vector of size equal to the move space.
        self.policy_head = nn.Linear(128, get_move_space_size())
        # Value head outputs a scalar evaluation.
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

##########################
# Replay Buffer Module   #
##########################
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

#################################
# Monte Carlo Tree Search (MCTS)#
#################################
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}    # move -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, agent, simulations=800, c_puct=1.0, dirichlet_alpha=0.3, epsilon=0.25, batch_size=16):
        self.agent = agent
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.batch_size = batch_size

    def search(self, board):
        root = MCTSNode(board)
        # Evaluate root state using the network
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            policy, value = self.agent.model(state_tensor)
        policy = policy.cpu().numpy()[0]
        legal_moves = list(board.legal_moves)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = self.agent.move_to_index(move)
            prior = (1 - self.epsilon) * policy[move_idx] + self.epsilon * noise[i]
            child_node = MCTSNode(board, parent=root)
            child_node.prior = prior
            root.children[move] = child_node

        # Run MCTS simulations with batched leaf evaluations
        pending_leaf_nodes = []
        pending_paths = []

        for _ in range(self.simulations):
            node = root
            search_path = [node]
            # SELECTION
            while node.children:
                best_score = -float('inf')
                best_move = None
                for move, child in node.children.items():
                    u = child.value() + self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
                    if u > best_score:
                        best_score = u
                        best_move = move
                node = node.children[best_move]
                search_path.append(node)
                if node.board.is_game_over():
                    break

            # If the node is non-terminal and needs evaluation, add to batch
            if not node.board.is_game_over():
                pending_leaf_nodes.append(node)
                pending_paths.append(search_path)
                if len(pending_leaf_nodes) >= self.batch_size:
                    self.evaluate_batch(pending_leaf_nodes, pending_paths)
                    pending_leaf_nodes = []
                    pending_paths = []
            else:
                # Terminal evaluation using game result
                result = node.board.result()
                leaf_value = 1 if result == "1-0" else -1 if result == "0-1" else 0
                self.backpropagate(search_path, leaf_value)

        # Process any remaining batched leaves
        if pending_leaf_nodes:
            self.evaluate_batch(pending_leaf_nodes, pending_paths)

        total_visits = sum(child.visit_count for child in root.children.values())
        move_probs = {move: child.visit_count / total_visits for move, child in root.children.items()}
        return move_probs

    def evaluate_batch(self, leaf_nodes, paths):
        # Prepare batch of state tensors
        states = [board_to_tensor(node.board) for node in leaf_nodes]
        batch_tensor = torch.stack(states).to(self.agent.device)
        with torch.no_grad():
            policies, values = self.agent.model(batch_tensor)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy().flatten()

        for node, path, policy, value in zip(leaf_nodes, paths, policies, values):
            legal_moves = list(node.board.legal_moves)
            node.children = {}
            for i, move in enumerate(legal_moves):
                move_idx = self.agent.move_to_index(move)
                child = MCTSNode(node.board, parent=node)
                child.prior = policy[move_idx]
                node.children[move] = child
            self.backpropagate(path, value)

    def backpropagate(self, search_path, leaf_value):
        # Backpropagate the evaluation value along the search path
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += leaf_value
            leaf_value = -leaf_value

#################################
# AlphaZero Agent for Chess RL  #
#################################
class ChessAlphaZeroAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = ChessNet().to(self.device)
        # Learning rate and optimizer can be tuned.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer()
        self.mcts_simulations = 800
        self.c_puct = 1.0

    def move_to_index(self, move):
        # Convert a chess.Move into an index.
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

    def select_move(self, board, temperature=1.0):
        # Use MCTS to select a move.
        mcts = MCTS(self, simulations=self.mcts_simulations, c_puct=self.c_puct)
        move_probs = mcts.search(board)
        moves = list(move_probs.keys())
        probs = np.array(list(move_probs.values()))
        # Adjust probabilities with temperature.
        if temperature != 1.0:
            probs = probs ** (1 / (temperature + 1e-10))
            probs = probs / np.sum(probs)
        chosen_move = np.random.choice(moves, p=probs)
        return chosen_move

    def self_play(self):
        """
        Play a game against itself using MCTS to guide moves.
        Save the board states, the MCTS-derived move probabilities,
        and, after game end, the outcome as training targets.
        """
        board = chess.Board()
        game_history = []
        while not board.is_game_over():
            move = self.select_move(board)
            state_tensor = board_to_tensor(board)
            # MCTS can be modified to also return move probabilities if needed.
            move_probs = {}  # placeholder if you want to record move distribution.
            game_history.append((state_tensor, move_probs))
            board.push(move)
        # Determine outcome: +1 for win, -1 for loss, 0 for draw.
        result = board.result()
        if result == "1-0":
            outcome = 1
        elif result == "0-1":
            outcome = -1
        else:
            outcome = 0
        # Store self-play transitions into the replay buffer.
        for state, pi in game_history:
            pi_tensor = torch.zeros(get_move_space_size())
            for move, prob in pi.items():
                idx = self.move_to_index(move)
                pi_tensor[idx] = prob
            value_tensor = torch.tensor([outcome], dtype=torch.float)
            self.replay_buffer.push(state, pi_tensor, value_tensor)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        states, policies, values = self.replay_buffer.sample(batch_size)
        pred_policies, pred_values = self.model(states)
        # If necessary, pad the target policy vector.
        if pred_policies.size(1) > policies.size(1):
            padding = torch.zeros(policies.size(0), pred_policies.size(1) - policies.size(1)).to(self.device)
            policies = torch.cat([policies, padding], dim=1)
        # Loss: cross-entropy for the policy head and MSE for the value head.
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
        value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
        total_loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def save_model(self, model_name="alphazero_chess_model.pth"):
        model_dir = os.path.join("model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))
        print(f"Model saved to '{model_dir}/{model_name}'")
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

