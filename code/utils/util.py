import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import names
import numpy as np
import random
import os
import chess

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

def board_to_feature_vector(board):
    """
    Convert a chess.Board object into a flat feature vector of length 363.
    The vector consists of:
      1. Basic features (1 + 4 + 10 = 15 features):
         - Side to move (1 feature)
         - Castling rights: white kingside, white queenside, black kingside, black queenside (4 features)
         - Material counts for queen, rook, bishop, knight, pawn for white and black (10 features)
      2. Piece list features (for both sides, 32 slots × 9 features = 288 features):
         For each side, reserve fixed slots for pieces in the following order:
            King: 1 slot
            Queen: 1 slot
            Rook: 2 slots
            Bishop: 2 slots
            Knight: 2 slots
            Pawn: 8 slots
         For each slot, if a piece is present, output 9 features:
            a. Presence flag (1.0 if present, else 0.0)
            b. Normalized file coordinate (file index divided by 7)
            c. Normalized rank coordinate (rank index divided by 7)
            d–g. Estimated mobility in 4 directions (up, down, left, right) computed as distance to edge / 7
            h–i. Two placeholder features (set to 0.0 for now) that you can later replace with attacker/defender info.
      3. Padding features (60 zeros) to bring total length to 363.
    """
    features = []
    
    # 1. Basic Features
    # 1.1 Side to move: 1 if white, 0 if black.
    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    # 1.2 Castling rights (order: white kingside, white queenside, black kingside, black queenside)
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    
    # 1.3 Material counts for queen, rook, bishop, knight, pawn for white and black.
    # Order: white queen, white rook, white bishop, white knight, white pawn,
    #        black queen, black rook, black bishop, black knight, black pawn.
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    for color in [chess.WHITE, chess.BLACK]:
        for p_type in piece_types:
            count = len(board.pieces(p_type, color))
            features.append(float(count))
    
    # 2. Piece List Features
    # For each side, we reserve fixed slots for pieces.
    # Define slots per piece type:
    piece_slots = {
        chess.KING: 1,
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
        chess.PAWN: 8
    }
    # Order for piece slots:
    piece_order = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    
    # Process white pieces first, then black pieces.
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in piece_order:
            slots = piece_slots[piece_type]
            # Get list of squares for this piece type and color, sorted by square index.
            squares = sorted(list(board.pieces(piece_type, color)))
            for i in range(slots):
                if i < len(squares):
                    # Piece is present.
                    features.append(1.0)  # presence flag
                    square = squares[i]
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    features.append(file / 7.0)  # normalized file
                    features.append(rank / 7.0)  # normalized rank
                    # Estimated mobility: simply the distance to each board edge.
                    # Up: distance to top (rank 7)
                    features.append((7 - rank) / 7.0)
                    # Down: distance to bottom (rank 0)
                    features.append(rank / 7.0)
                    # Left: distance to file 0
                    features.append(file / 7.0)
                    # Right: distance to file 7
                    features.append((7 - file) / 7.0)
                    # Two placeholder features for additional info (e.g., attacker/defender scores)
                    features.append(0.0)
                    features.append(0.0)
                else:
                    # No piece in this slot: fill with zeros (9 features per slot).
                    features.extend([0.0] * 9)
    
    # 3. Padding: append 60 zeros to reach total length 363.
    features.extend([0.0] * 60)
    
    feature_vector = np.array(features, dtype=np.float32)
    # Safety check: ensure the vector has length 363.
    if feature_vector.shape[0] != 363:
        raise ValueError(f"Feature vector length is {feature_vector.shape[0]}, but expected 363.")
    return feature_vector

def get_feature_vector_size():
    """Return the size of the feature vector produced by board_to_feature_vector."""
    return 363
