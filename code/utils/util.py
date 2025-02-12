import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import names

def get_move_space_size():
    """Return the total size of the move space."""
    regular_moves = 64 * 64
    promotion_moves_per_piece = 64 * 8
    promotion_pieces = 4  # Queen, Rook, Bishop, Knight
    promotion_moves = promotion_moves_per_piece * promotion_pieces
    return regular_moves + promotion_moves

def board_to_tensor(board):
    """Convert chess board to tensor representation"""
    pieces = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
             'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    tensor = torch.zeros(8, 8, 8)
    
    # Set piece planes
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = i // 8, i % 8
            piece_idx = pieces[piece.symbol()]
            color_idx = 0 if piece.color else 1
            plane_idx = piece_idx // 2
            tensor[plane_idx][rank][file] = 1 if color_idx == 0 else -1
            
    # Add repetition planes
    tensor[6] = torch.ones(8, 8) if board.turn else torch.zeros(8, 8)
    tensor[7] = torch.ones(8, 8) * len(board.move_stack) / 100.0
    
    return tensor.permute(0, 1, 2)

def generate_unique_model_name(prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{names.get_first_name()}{names.get_first_name()}{names.get_first_name()}_{timestamp}"