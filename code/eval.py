import argparse
import chess
import torch
import os
import json
import datetime
import io
import zlib
import psutil

from rl_agent.agent_neural import ChessNeuralAgent
from utils.util import board_to_tensor, get_move_space_size

# Import the Stockfish agent for evaluation.
from built_in_agent.agent_stockfish import ChessStockfishAgent

def evaluate_game(neural_agent, competitor):
    """
    Play a single game between neural_agent and competitor.
    The neural_agent will play as White in half of the games and as Black in the other half.
    
    Returns:
        result: 'win' if neural_agent wins, 'loss' if it loses, or 'draw'
    """
    board = chess.Board()
    # Randomly decide which side the neural agent will play.
    neural_agent_is_white = (torch.rand(1).item() > 0.5)
    
    while not board.is_game_over():
        if board.turn == neural_agent_is_white:
            # Neural agent’s turn (select greedily with temperature=0).
            move = neural_agent.select_move(board, temperature=0)
        else:
            # Opponent’s turn. No time limit is enforced.
            move = competitor.select_move(board)
        board.push(move)
    
    result_str = board.result()  # e.g., "1-0", "0-1", or "1/2-1/2"
    if result_str == "1-0":
        return "win" if neural_agent_is_white else "loss"
    elif result_str == "0-1":
        return "win" if not neural_agent_is_white else "loss"
    else:
        return "draw"

def run_evaluation(neural_agent, competitor, num_games=50):
    wins = 0
    losses = 0
    draws = 0

    for game in range(1, num_games + 1):
        result = evaluate_game(neural_agent, competitor)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1
        print(f"Game {game}: {result.upper()}")
    
    print("\nEvaluation Summary:")
    print(f"Total games: {num_games}")
    print(f"Wins: {wins}  Draws: {draws}  Losses: {losses}")
    win_rate = wins / num_games * 100.0
    print(f"Win rate: {win_rate:.2f}%")

    return win_rate

def show_computational_metrics(model):
    """
    Display computational metrics for the model:
    - Estimated RAM usage (based on parameter count assuming 32-bit floats)
    - Uncompressed and compressed model size (using zlib for compression)
    - Dedicated CPU requirement (as per given constraints)
    """
    # Estimate RAM usage: each parameter is assumed to be a 32-bit float (4 bytes).
    param_count = sum(p.numel() for p in model.model.parameters())
    estimated_ram_bytes = param_count * 4
    estimated_ram_mib = estimated_ram_bytes / (1024 ** 2)
    print(f"Estimated model RAM usage (parameters): {estimated_ram_mib:.2f} MiB")
    
    # Save state_dict to a bytes buffer and measure its size.
    buffer = io.BytesIO()
    torch.save(model.model.state_dict(), buffer)
    uncompressed_size_kib = buffer.tell() / 1024
    # Compress the state_dict to measure compressed size.
    compressed_data = zlib.compress(buffer.getvalue())
    compressed_size_kib = len(compressed_data) / 1024
    
    print(f"Uncompressed model size: {uncompressed_size_kib:.2f} KiB")
    print(f"Compressed model size: {compressed_size_kib:.2f} KiB")

    cpu_freq = psutil.cpu_freq()
    print(f"Current CPU frequency: {cpu_freq.current:.2f} MHz")
    print(f"Min CPU frequency: {cpu_freq.min:.2f} MHz")
    print(f"Max CPU frequency: {cpu_freq.max:.2f} MHz")

    return estimated_ram_mib, compressed_size_kib, cpu_freq
    
    
    # Note: The constraints mention 5 MiB of RAM and 64KiB submission size limit.
    # These printed values can help gauge if your model meets these resource constraints.

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the performance of a neural chess agent against a competitor."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=50,
        help="Number of evaluation games to play."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained neural agent model checkpoint."
    )
    parser.add_argument(
        "--competitor",
        type=str,
        default="stockfish",
        choices=["stockfish", "model"],
        help="Type of competitor to use: 'stockfish' or 'model'."
    )
    parser.add_argument(
        "--competitor_model_path",
        type=str,
        help="Path to the competitor's neural model checkpoint (required if competitor is 'neural')."
    )
    args = parser.parse_args()

    # Load configuration from config.json for engine paths.
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Optionally, print a timestamp for the evaluation run.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting evaluation run at {timestamp}")

    # Instantiate the neural agent and load its checkpoint.
    neural_agent = ChessNeuralAgent()
    neural_agent.load_model(args.model_path)
    neural_agent.eval()  # Set model to evaluation mode

    # Display computational metrics for the neural agent.
    print("\n--- Model Computational Metrics ---")
    estimated_ram_mib, compressed_size_kib, cpu_freq = show_computational_metrics(neural_agent)
    print("-------------------------------------\n")

    # Instantiate the competitor.
    if args.competitor.lower() == "model":
        if not args.competitor_model_path:
            raise ValueError("Competitor model path must be provided when competitor is 'neural'.")
        competitor = ChessNeuralAgent()
        competitor.load_model(args.competitor_model_path)
        competitor.eval()
    elif args.competitor.lower() == "stockfish":
        stockfish_path = config.get("engine", {}).get("stockfish_path", "")
        if not stockfish_path:
            raise ValueError("Stockfish engine path must be specified in config.json under engine.stockfish_path.")
        competitor = ChessStockfishAgent(engine_path=stockfish_path)
    else:
        raise ValueError("Invalid competitor type specified. Must be either 'stockfish' or 'neural'.")

    # Run evaluation games.
    win_rate = run_evaluation(neural_agent, competitor, args.num_games)

if __name__ == "__main__":
    main()
