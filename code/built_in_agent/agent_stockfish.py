import chess
import chess.engine

class ChessStockfishAgent:
    def __init__(self, engine_path, time_limit=0.1):
        """
        Initialize the Stockfish agent.
        
        :param engine_path: The full path to the Stockfish executable.
        :param time_limit: The time (in seconds) Stockfish is allowed to think per move.
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.time_limit = time_limit

    def select_move(self, board):
        """
        Select a move for the given board position.
        
        :param board: A chess.Board object representing the current position.
        :return: A chess.Move object representing the move chosen by Stockfish.
        """
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move

    def quit(self):
        """
        Shut down the Stockfish engine.
        """
        self.engine.quit()

