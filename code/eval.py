import argparse
import chess
import torch
import os
import json
import datetime
import io
import zlib
import psutil
import logging
import time
import secrets
import string
import pandas as pd
import re
from typing import List, Dict

from rl_agent.agent_neural import ChessNeuralAgent
from utils.util import board_to_tensor, get_move_space_size

# Import the Stockfish agent for evaluation.
from built_in_agent.agent_stockfish import ChessStockfishAgent
from rl_agent.agent_giraffe import GiraffeChessAgent

# ---------------------------------------------------------------------------
# Game evaluation functions
# ---------------------------------------------------------------------------
def evaluate_game(rl_agent, competitor):
    """
    Play a single game between rl_agent and competitor.
    The rl_agent will play as White in half of the games and as Black in the other half.
    
    Returns:
        result: 'win' if rl_agent wins, 'loss' if it loses, or 'draw'
    """
    board = chess.Board()
    # Randomly decide which side the RL agent will play.
    rl_agent_is_white = (torch.rand(1).item() > 0.5)
    
    while not board.is_game_over():
        if board.turn == rl_agent_is_white:
            move, _ = rl_agent.select_move(board, temperature=0)
        else:
            move, _ = competitor.select_move(board)
        board.push(move)
    
    result_str = board.result()  # e.g., "1-0", "0-1", or "1/2-1/2"
    if result_str == "1-0":
        return "win" if rl_agent_is_white else "loss"
    elif result_str == "0-1":
        return "win" if not rl_agent_is_white else "loss"
    else:
        return "draw"

def run_evaluation(rl_agent, competitor, num_games=50):
    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        # Update game number (Game 0, Game 1, Game 2, ...)
        _ = game  # The game number is updated but not printed.
        result = evaluate_game(rl_agent, competitor)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1
        print(f"Game {game}: {result}")
    
    print("\nEvaluation Summary:")
    print(f"Total games: {num_games}")
    print(f"Wins: {wins}  Draws: {draws}  Losses: {losses}")
    win_rate = wins / num_games * 100.0
    draw_rate = draws / num_games * 100.0
    loss_rate = losses / num_games * 100.0
    print(f"Win rate: {win_rate:.2f}%  Draw rate: {draw_rate:.2f}%  Loss rate: {loss_rate:.2f}%")

    return win_rate, draw_rate, loss_rate

def show_computational_metrics(model):
    """
    Display computational metrics for the model:
    - Estimated RAM usage (based on parameter count assuming 32-bit floats)
    - Uncompressed and compressed model size (using zlib for compression)
    - CPU frequency details
    """
    param_count = sum(p.numel() for p in model.model.parameters())
    estimated_ram_bytes = param_count * 4
    estimated_ram_mib = estimated_ram_bytes / (1024 ** 2)
    print(f"Estimated model RAM usage (parameters): {estimated_ram_mib:.2f} MiB")
    
    buffer = io.BytesIO()
    torch.save(model.model.state_dict(), buffer)
    uncompressed_size_kib = buffer.tell() / 1024
    compressed_data = zlib.compress(buffer.getvalue())
    compressed_size_kib = len(compressed_data) / 1024
    
    print(f"Uncompressed model size: {uncompressed_size_kib:.2f} KiB")
    print(f"Compressed model size: {compressed_size_kib:.2f} KiB")

    cpu_freq = psutil.cpu_freq()
    print(f"Current CPU frequency: {cpu_freq.current:.2f} MHz")
    print(f"Min CPU frequency: {cpu_freq.min:.2f} MHz")
    print(f"Max CPU frequency: {cpu_freq.max:.2f} MHz")

    return estimated_ram_mib, compressed_size_kib, cpu_freq

# ---------------------------------------------------------------------------
# Strategic Test Suite (STS) evaluation functions (using EPD files)
# ---------------------------------------------------------------------------

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


def evaluate_strategic_test_suite(rl_agent, epd_file_path="data/STS1-STS15_LAN_v3.epd"):
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
        
        try:
            selected_move, _ = rl_agent.select_move(board, temperature=0)
        except Exception as e:
            logging.error("Error selecting move for test case %d: %s", idx, str(e))
            continue
        
        selected_uci = selected_move.uci()
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


def get_best_model(folder="model_best"):
    # List all .pth files in the folder.
    pth_files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in folder {folder}")
    # Optionally, choose the most recently modified .pth file.
    best_checkpoint = max(pth_files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    checkpoint_path = os.path.join(folder, best_checkpoint)
    return checkpoint_path
# ---------------------------------------------------------------------------
# Main function integrating both evaluations
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a neural chess agent against Stockfish (or a competitor model) and run a Strategic Test Suite (STS) evaluation."
    )
    # Game evaluation arguments
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of evaluation games to play.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained neural agent model checkpoint.")
    parser.add_argument("--competitor_model_path", type=str, default=get_best_model("model_best"),
                        help="Path to the competitor's neural model checkpoint. If provided, it is used as competitor; otherwise, Stockfish is used.")
    # STS evaluation argument (required)
    parser.add_argument("--sts-epd-file", type=str, required=False, default="data/STS1-STS15_LAN_v3.epd",
                        help="Path to the EPD file containing STS positions.")
    
    args = parser.parse_args()

    # Load configuration from config.json for engine paths (for Stockfish).
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Instantiate the neural agent and load its checkpoint.
    rl_agent = ChessNeuralAgent()
    rl_agent = GiraffeChessAgent()
    rl_agent.load_model(args.model_path)
    rl_agent.eval()  # Set model to evaluation mode

    # Run Strategic Test Suite (STS) evaluation.
    print("\nStarting Strategic Test Suite (STS) evaluation ---")
    sts_total, sts_percentage = evaluate_strategic_test_suite(rl_agent, args.sts_epd_file)
    print(f"Final STS Score: {sts_total} ({sts_percentage:.2f}%)")
    print("-------------------------------------\n")

    print("\n--- Model Computational Metrics ---")
    estimated_ram_mib, compressed_size_kib, cpu_freq = show_computational_metrics(rl_agent)
    print("-------------------------------------\n")

    # Load competitor
    competitor = ChessNeuralAgent()
    competitor.load_model(args.competitor_model_path)
    competitor.eval()
    print("\n--- Starting game evaluation (RL agent vs competitor) ---")
    run_evaluation(rl_agent, competitor, args.num_games)
    print("-------------------------------------\n")

    stockfish_path = config.get("engine", {}).get("stockfish_path", "")
    if not stockfish_path:
        raise ValueError("Stockfish engine path must be specified in config.json under engine.stockfish_path.")
    competitor = ChessStockfishAgent(engine_path=stockfish_path)

    # Run game evaluation: RL agent vs StockFish.
    print("\n--- Starting game evaluation (RL agent vs StockFish) ---")
    run_evaluation(rl_agent, competitor, args.num_games)
    print("-------------------------------------\n")


if __name__ == "__main__":
    main()
