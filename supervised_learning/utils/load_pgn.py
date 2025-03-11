import os
from chess import pgn
from torch.utils.data import IterableDataset
import torch
import numpy as np
from tqdm import tqdm

def load_pgn_games(file_path):
    """
    Generator that yields games from a PGN file.
    """
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game

def build_move_mapping(pgn_dir, limit_files=28):
    """
    Scans PGN files to build a unique mapping of move strings to integers.
    """
    unique_moves = set()
    files = [file for file in os.listdir(pgn_dir) if file.endswith(".pgn")]
    files = files[:min(len(files), limit_files)]
    for file in tqdm(files, desc="Scanning PGN files for move mapping"):
        file_path = os.path.join(pgn_dir, file)
        for game in load_pgn_games(file_path):
            board = game.board()
            for move in game.mainline_moves():
                unique_moves.add(move.uci())
                board.push(move)
    move_to_int = {move: idx for idx, move in enumerate(sorted(unique_moves))}
    return move_to_int

def board_to_matrix(board):
    """
    Convert a chess.Board instance into a 13x8x8 numpy matrix.
    """
    import numpy as np
    matrix = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        offset = 0 if piece.color else 6
        matrix[piece_type + offset, row, col] = 1
    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        matrix[12, row, col] = 1
    return matrix

class PGNIterableDataset(IterableDataset):
    """
    An IterableDataset that lazily reads PGN files, converts board states,
    and yields (input, label) pairs without loading everything in memory.
    """
    def __init__(self, pgn_dir, move_to_int, limit_files=28):
        super(PGNIterableDataset, self).__init__()
        self.pgn_dir = pgn_dir
        self.move_to_int = move_to_int
        self.limit_files = limit_files
        self.files = [os.path.join(pgn_dir, file) 
                      for file in os.listdir(pgn_dir) if file.endswith(".pgn")]
        self.files = self.files[:min(len(self.files), limit_files)]
    
    def __iter__(self):
        for file_path in self.files:
            for game in load_pgn_games(file_path):
                board = game.board()
                for move in game.mainline_moves():
                    matrix = board_to_matrix(board)
                    label = self.move_to_int[move.uci()]
                    yield torch.tensor(matrix, dtype=torch.float32), label
                    board.push(move)
