import torch
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, num_classes=4672): # The number 4672 is derived by enumerating all potential moves from each square on an 8×8 board—including promotions (with all options), castling, and underpromotions—for every piece type.
        super(Agent, self).__init__()
        # Define the network architecture.
        num_classes = 4672
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
        # Initialize weights.
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def board_to_matrix(self, board):
        """
        Convert a chess.Board instance into a 13x8x8 numpy matrix.
        The first 12 channels represent pieces; the 13th marks legal moves.
        """
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

    def prepare_input(self, board):
        """
        Convert board state to tensor input.
        """
        matrix = self.board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return X_tensor

    def predict_move(self, board, device, int_to_move):
        """
        Given a board, predict the best legal move using the model.
        """
        self.eval()  # Ensure the model is in evaluation mode.
        X_tensor = self.prepare_input(board).to(device)
        with torch.no_grad():
            logits = self.forward(X_tensor)
        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        sorted_indices = np.argsort(probabilities)[::-1]
        legal_moves = [move.uci() for move in board.legal_moves]
        for move_index in sorted_indices:
            move = int_to_move[move_index]
            if move in legal_moves:
                return move
        return None
