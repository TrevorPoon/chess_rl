import math
import chess
import torch
import numpy as np
from copy import deepcopy
from utils.util import board_to_tensor, get_move_space_size

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior  # P(s,a) from the neural network
        self.children = {}  # map of move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.state = None  # Store the board tensor representation
        
    def is_leaf(self):
        return len(self.children) == 0
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
        
    def select_child(self, c_puct=1.0):
        """Select the child with the highest UCB score."""
        best_score = float('-inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # UCB score = Q + U, where U ~ P * sqrt(N) / (1 + n)
            q_value = -child.value()  # Negative because value is from opponent's perspective
            u_value = (c_puct * child.prior * 
                      math.sqrt(self.visit_count) / (1 + child.visit_count))
            ucb_score = q_value + u_value

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
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
                
        
    def search(self, board):
        """Perform MCTS search starting from the given board position."""
        root = MCTSNode(board)
        
        # Evaluate the root state with the neural network
        root.state = board_to_tensor(root.board).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            policy, value = self.model(root.state)
        policy = policy.squeeze().cpu().numpy()
        value = value.item()
        
        # Create children for all legal moves
        for move in root.board.legal_moves:
            move_idx = self.move_to_index(move)
            child_board = root.board.copy()
            child_board.push(move)
            root.children[move] = MCTSNode(
                child_board,
                parent=root,
                prior=policy[move_idx]
            )
        
        # Perform simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf node
            while not node.is_leaf() and not node.board.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.board.is_game_over():
                node.state = board_to_tensor(node.board).unsqueeze(0).to('cuda:0')
                with torch.no_grad():
                    policy, value = self.model(node.state)
                policy = policy.squeeze().cpu().numpy()
                value = value.item()
                
                # Create children for all legal moves
                for move in node.board.legal_moves:
                    move_idx = self.move_to_index(move)
                    child_board = node.board.copy()
                    child_board.push(move)
                    node.children[move] = MCTSNode(
                        child_board,
                        parent=node,
                        prior=policy[move_idx]
                    )
            else:
                # Game is over, use the game result as value
                if node.board.is_checkmate():
                    value = -1 if node.board.turn else 1
                else:
                    value = 0  # Draw
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value  # Value from opponent's perspective
        
        # Return policy vector based on visit counts
        policy = np.zeros(1968)  # Maximum possible moves
        for move, child in root.children.items():
            policy[self.move_to_index(move)] = child.visit_count
        
        # Temperature-adjusted policy
        policy = policy ** (1/self.temperature)
        policy = policy / np.sum(policy)
        
        return policy

class ChessRLWithMCTS():
    def __init__(self, num_simulations=800):
        super().__init__()
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
    
    def select_move(self, board, temperature=1.0):
        """Select a move using MCTS."""
        self.mcts.temperature = temperature
        policy = self.mcts.search(board)
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        move_probs = np.array([policy[self.mcts.move_to_index(move)] for move in legal_moves])
        
        # Select move based on policy
        move_idx = np.random.choice(len(legal_moves), p=move_probs/np.sum(move_probs))
        return legal_moves[move_idx]
