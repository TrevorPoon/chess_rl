import chess
import chess.engine
import torch
import numpy as np
from rl_agent.agent_neural import ChessNeuralAgent
from utils.util import board_to_tensor, get_move_space_size

class ChessDistillationAgent(ChessNeuralAgent):
    def __init__(self, stockfish_path, engine_depth=15):
        """
        Initialize the distillation agent.
        
        Args:
            stockfish_path (str): Path to the Stockfish binary.
            engine_depth (int): Depth for Stockfish analysis.
        """
        super(ChessDistillationAgent, self).__init__()
        self.engine_depth = engine_depth
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print("Stockfish engine started successfully for distillation.")
        except Exception as e:
            print(f"Error starting Stockfish engine: {e}")
            raise e

    def board_to_teacher_policy(self, board, teacher_move):
        """
        Convert Stockfish's teacher move into a one-hot encoded policy vector.
        
        Args:
            board (chess.Board): The current board state.
            teacher_move (chess.Move): The move suggested by Stockfish.
            
        Returns:
            torch.Tensor: A one-hot encoded tensor representing the teacher's move.
        """
        move_space_size = get_move_space_size()
        teacher_policy = torch.zeros(move_space_size)
        move_idx = self.move_to_index(teacher_move)
        teacher_policy[move_idx] = 1.0
        return teacher_policy

    def board_to_teacher_value(self, info):
        """
        Convert Stockfish's evaluation info into a value target.
        The evaluation is scaled via tanh to roughly lie between -1 and 1.
        
        Args:
            info (dict): The analysis dictionary returned by Stockfish.
            
        Returns:
            torch.Tensor: A tensor containing the scaled evaluation value.
        """
        if "score" in info:
            # Get score from White's perspective. Use mate_score as fallback.
            score = info["score"].white().score(mate_score=10000)
            value = np.tanh(score / 1000.0)
        else:
            value = 0.0
        return torch.tensor([value], dtype=torch.float32)

    def collect_teacher_data(self, max_moves=100):
        """
        Play a game against Stockfish to collect teacher (distillation) data.
        The agent plays as White (collecting teacher moves and evaluations), and Stockfish
        plays as Black.
        
        Args:
            max_moves (int): Maximum number of moves to simulate in the game.
        
        Returns:
            int: The number of experiences collected.
        """
        board = chess.Board()
        experiences = []
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                try:
                    # Query Stockfish for its best move and evaluation.
                    result = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth))
                except Exception as e:
                    print(f"Engine analysis error: {e}")
                    break

                teacher_move = result["pv"][0]
                teacher_policy = self.board_to_teacher_policy(board, teacher_move)
                teacher_value = self.board_to_teacher_value(result)
                state_tensor = board_to_tensor(board)
                experiences.append((state_tensor, teacher_policy, teacher_value))
                # For distillation, we follow the teacher's move.
                board.push(teacher_move)
            else:
                try:
                    # Let Stockfish make its move when playing as Black.
                    result = self.engine.play(board, chess.engine.Limit(depth=self.engine_depth))
                    board.push(result.move)
                except Exception as e:
                    print(f"Engine play error: {e}")
                    break
            move_count += 1

        # Push all collected experiences into the replay buffer.
        for exp in experiences:
            self.replay_buffer.push(*exp)

        print(f"Collected {len(experiences)} teacher experiences.")
        return len(experiences)

    def close_teacher(self):
        """
        Cleanly shutdown the Stockfish engine.
        """
        self.engine.quit()
        print("Stockfish engine closed for distillation.")
