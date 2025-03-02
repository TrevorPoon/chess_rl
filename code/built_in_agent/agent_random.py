import chess
import random

class ChessRandomAgent:
    def __init__(self):
        """
        Initialize the Random Agent.
        """
        pass  # No initialization needed for a random agent

    def select_move(self, board):
        """
        Select a random move for the given board position.

        :param board: A chess.Board object representing the current position.
        :return: A chess.Move object representing a randomly chosen legal move.
        """
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

    def quit(self):
        """
        Shut down the Stockfish engine.
        """
        self.engine.quit()

# Example usage:
if __name__ == "__main__":
    board = chess.Board()
    random_agent = ChessRandomAgent()
    move = random_agent.select_move(board)
    print("Random move selected:", move)
