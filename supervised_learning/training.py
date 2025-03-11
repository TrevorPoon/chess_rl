import os
import time
import pickle
import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from chess import pgn, Board
from typing import List, Dict
import logging

#########################################
#         Helper Functions              #
#########################################

def board_to_matrix(board: Board):
    """
    Convert a chess.Board instance into a 13x8x8 numpy matrix.
    The first 12 channels represent the pieces (6 types for each color).
    The 13th channel marks the legal moves destination squares.
    """
    matrix = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    # Populate first 12 channels with piece locations.
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        offset = 0 if piece.color else 6
        matrix[piece_type + offset, row, col] = 1

    # Populate the legal moves channel (13th channel).
    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        matrix[12, row, col] = 1

    return matrix

def create_input_for_nn(games):
    """
    For each game, convert board states to matrix form and record the move labels.
    """
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)

def encode_moves(moves):
    """
    Encode moves (strings) as integers.
    """
    unique_moves = set(moves)
    move_to_int = {move: idx for idx, move in enumerate(unique_moves)}
    encoded = np.array([move_to_int[move] for move in moves], dtype=np.int64)
    return encoded, move_to_int

def prepare_input(board: Board):
    """
    Prepare board input for the neural network prediction.
    """
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

#########################################
#           Dataset Class               #
#########################################

class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#########################################
#           Model Definition            #
#########################################

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits output
        return x

#########################################
#           PGN Loading Function        #
#########################################

def load_pgn(file_path):
    """
    Load games from a PGN file.
    """
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

#########################################
#           Prediction Function         #
#########################################

def predict_move(board: Board, model, device, int_to_move):
    """
    Given a board, use the model to predict the best legal move.
    """
    X_tensor = prepare_input(board).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    logits = logits.squeeze(0)  # Remove batch dimension
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    
    sorted_indices = np.argsort(probabilities)[::-1]
    legal_moves = [move.uci() for move in board.legal_moves]
    
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves:
            return move
    return None

#########################################
#       Evaluation
#########################################

def read_epd_file(epd_path: str) -> List[Dict[str, object]]:
    """
    Reads an EPD file and returns a list of test case dictionaries.
    
    Each test case dictionary contains:
      - "fen": the board position in FEN notation.
      - "expected_moves": a dict mapping solution move strings to their scores.
      
    The EPD file is expected to include fields:
      - "bm": best move(s) with a score of 10.
      - "am": alternative move(s) with a default score of 5.
      
    Lines that are empty, start with '#' or cannot be parsed correctly are skipped.
    """
    def parse_epd_line(line: str) -> Dict[str, object]:
        """
        Parses one EPD line and returns a dictionary with FEN and expected moves.

        The FEN is taken from the first four space-separated tokens, and we append 
        default halfmove and fullmove counters ("0 1").
        The "bm" (best move) and "am" (alternative move) fields are searched for in the 
        remaining parts of the line. Moves in the bm field get a score of 10, and moves 
        in the am field get a score of 5 if not already assigned.
        
        Returns a dictionary with keys:
          - "fen": a string containing the FEN.
          - "expected_moves": a dict mapping move strings to their score.
        
        If the line is not valid (e.g. not enough tokens or no move fields found), returns an empty dict.
        """
        # Split the line on ';' to separate the main part from any comments.
        parts = line.split(';')
        if not parts:
            return {}
        
        # Split the first part into tokens.
        fen_tokens = parts[0].split()
        # Expect at least 4 tokens for our FEN; if not, skip this line.
        if len(fen_tokens) < 4:
            return {}
        # Construct the FEN using the first 4 tokens, and add default halfmove and fullmove counts.
        fen = " ".join(fen_tokens[:4]) + " 0 1"
        
        expected_moves = {}
        notation_moves = []
        scores = []
        for part in parts[1:]:
            part = part.strip()
            if part.startswith("c9"):
                # Remove the tag "c9" and any double quotes, then split on whitespace.
                cleaned = part.replace('c9', '').replace('"', '').strip()
                notation_moves = cleaned.split()
            elif part.startswith("c8"):
                cleaned = part.replace('c8', '').replace('"', '').strip()
                scores = [int(score) for score in cleaned.split()]

        for i in range(len(notation_moves)):
            move = notation_moves[i]
            expected_moves[move] = scores[i]

        if not expected_moves:
            return {}

        return {"fen": fen, "expected_moves": expected_moves}
    
    test_cases = []
    with open(epd_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines starting with '#'
            if not line or line.startswith("#"):
                continue
            case = parse_epd_line(line)
            if case:
                test_cases.append(case)
    return test_cases

def evaluate_strategic_test_suite(model, device, int_to_move, epd_file_path="data/STS1-STS15_LAN_v3.epd"):
    """
    Evaluate the neural agent's performance on a Strategic Test Suite (STS) loaded from an EPD file.
    
    For each test case, the agent selects a move (using greedy selection with temperature=0). 
    The selected move is compared against the expected moves (from the "bm" and "am" fields) and awarded a score:
      - 10 if the move is in the "bm" field,
      - 5 if it is only in the "am" field,
      - 0 otherwise.
    
    The final score is the sum of scores for all positions, with a maximum of 10 per position.
    For example, for a suite of 1500 positions, the maximum possible score is 15000.
    
    Returns:
        A tuple (total_score, percentage) where:
          - total_score is the sum of scores,
          - percentage is the percentage of the maximum possible score.
    """
    test_cases = read_epd_file(epd_file_path)
    total_score = 0
    max_score = len(test_cases) * 10  # each position has a maximum score of 10

    logging.info("Starting STS evaluation on %d positions", len(test_cases))
    start_time = time.perf_counter()
    
    for idx, test in enumerate(test_cases, start=1):
        fen = test.get("fen")
        expected_moves = test.get("expected_moves", {})  # dict: move -> score
        board = chess.Board(fen)
        
        move_uci = predict_move(board, model, device, int_to_move)
        
        selected_uci = move_uci.uci()
        score_awarded = expected_moves.get(selected_uci, 0)
        total_score += score_awarded
        logging.info("Test %d: Selected move %s, Expected: %s, Score: %d",
                     idx, selected_uci, expected_moves, score_awarded)
    
    overall_elapsed = time.perf_counter() - start_time
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    logging.info("STS Evaluation: Total score %d/%d (%.2f%%) over %.2f sec",
                 total_score, max_score, percentage, overall_elapsed)
    
    print(f"\nSTS Evaluation: Total score {total_score}/{max_score} ({percentage:.2f}%)")
    return total_score, percentage

def evaluate_random_model(model, device, int_to_move, num_games=100):
    """
    Have the trained model (playing as White) face a random agent (playing as Black)
    for a given number of games. Return wins, draws, and losses.
    """
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        board = Board()
        # White is model, Black is random
        while not board.is_game_over():
            if board.turn:  # White's turn: use model prediction
                move_uci = predict_move(board, model, device, int_to_move)
                # If no legal move predicted, choose randomly
                if move_uci is None:
                    move = random.choice(list(board.legal_moves))
                else:
                    import chess
                    move = chess.Move.from_uci(move_uci)
            else:  # Black's turn: random move
                move = random.choice(list(board.legal_moves))
            board.push(move)
        result = board.result()  # "1-0", "0-1", or "1/2-1/2"
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
    return wins, draws, losses

#########################################
#              Training Code            #
#########################################

def train_model():
    # Load PGN files
    pgn_dir = os.path.join("data", "pgn")
    files = [file for file in os.listdir(pgn_dir) if file.endswith(".pgn")]
    LIMIT_OF_FILES = min(len(files), 28)
    
    all_games = []
    file_counter = 0
    for file in tqdm(files, desc="Loading PGN files"):
        file_path = os.path.join(pgn_dir, file)
        all_games.extend(load_pgn(file_path))
        file_counter += 1
        if file_counter >= LIMIT_OF_FILES:
            break

    # Create input and output arrays
    X, y = create_input_for_nn(all_games)
    print(f"NUMBER OF SAMPLES: {len(y)}")
    
    # Encode moves and convert arrays to tensors
    y_encoded, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # Create Dataset and DataLoader
    dataset = ChessDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Initialize model, criterion, and optimizer
    model = ChessModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Prepare CSV file for evaluation results
    csv_filename = "evaluation_results.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Wins", "Draws", "Losses", "STS Score", "STS Percentage"])
    
    num_epochs = 100
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Time: {minutes}m {seconds}s")
        
        # Evaluate the model against a random agent for 100 games after each epoch.
        wins, draws, losses = evaluate_random_model(model, device, {v: k for k, v in move_to_int.items()}, num_games=100)
        print(f"Evaluation after Epoch {epoch+1}: Wins: {wins}, Draws: {draws}, Losses: {losses}")

        sts_score, sts_percentage = evaluate_strategic_test_suite(model, device, {v: k for k, v in move_to_int.items()})
        print(f"STS Score: {sts_score}, STS Percentage: {sts_percentage}")
        
        # Append evaluation results to CSV file.
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch+1, wins, draws, losses, sts_score, sts_percentage])
    
        # Save the model and the move encoding mapping
        model_path = os.path.join("model", "TORCH_100EPOCHS.pth")
        pickle_path = os.path.join("model", "move_to_int.pkl")
        torch.save(model.state_dict(), model_path)
        with open(pickle_path, "wb") as file:
            pickle.dump(move_to_int, file)
    print("Training complete. Model and mapping saved.")

#########################################
#          Inference / Demo Code        #
#########################################

def demo_prediction():
    # Load the move mapping
    pickle_path = os.path.join("model", "move_to_int.pkl")
    with open(pickle_path, "rb") as file:
        move_to_int = pickle.load(file)
    int_to_move = {v: k for k, v in move_to_int.items()}
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Initialize and load the model
    num_classes = len(move_to_int)
    model = ChessModel(num_classes=num_classes)
    model_path = os.path.join("model", "TORCH_100EPOCHS.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create an example chess board (starting position)
    board = Board()
    predicted_move = predict_move(board, model, device, int_to_move)
    print(f"Predicted move for the starting position: {predicted_move}")

#########################################
#           Main Entry Point            #
#########################################

if __name__ == '__main__':
    # Uncomment one of the following to either train the model or run a demo prediction.
    
    # To train the model, uncomment the next line:
    # train_model()
    
    # To run a demo prediction, uncomment the next line:
    # demo_prediction()
    
    # For this example, we'll run the training routine.
    train_model()
