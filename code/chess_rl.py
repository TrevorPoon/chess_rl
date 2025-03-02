import argparse
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import wandb
from datetime import datetime
import gc

from rl_agent.agent_neural import ChessNeuralAgent
from rl_agent.agent_mcts import ChessAlphaZeroAgent
from rl_agent.agent_distill import ChessDistillationAgent
from rl_agent.agent_light import ChessLightAgent
from built_in_agent.agent_stockfish import ChessStockfishAgent


from utils.chess_recorder import ChessVideoRecorder
from utils.util import *
from eval import run_evaluation, show_computational_metrics, evaluate_strategic_test_suite




BEST_MODEL_PATH = "model_best/Self-play_20250223_193946_neural_fanciful-mountain-17.pth"

def self_play_training(model, num_games=1000000, moves_per_game=10000, viz_every=1000):
    """Train the model through self-play with video recording."""
    recorder = ChessVideoRecorder()

    # Use the updated wandb.run.name (random name with timestamp appended)
    model_filename = wandb.run.name
    
    for game in range(num_games):
        board = chess.Board()
        game_states = []
        should_record = args.record and (game % viz_every == 0)
        
        if should_record:
            recorder.start_game(game)
            recorder.save_frame(board)
        
        for move_num in range(moves_per_game):
            if board.is_game_over():
                break
            
            # Store current state
            state = board_to_tensor(board)
            
            # Select and make move using the training model
            temperature = max(1.0 - move_num / 30, 0.1)
            move = model.select_move(board, temperature)
            board.push(move)
            
            # Store state and move
            game_states.append((state, move))
            
            # Record position if we're recording this game
            if should_record:
                recorder.save_frame(board)
        
        # Finish recording if we were recording this game
        if should_record:
            video_name = f"{model_filename}"
            recorder.end_game(video_name, framerate=2)
            print(f"\nVideo saved for game {game} as {video_name}!")
        
        # Determine game outcome
        if board.is_checkmate():
            # In self-play, the current board.turn indicates the player who is about to move
            value = 1.0 if board.turn == False else -1.0
            outcome = "Checkmate"
        elif board.is_stalemate():
            value = 0.0
            outcome = "Stalemate"
        elif board.is_insufficient_material():
            value = 0.0
            outcome = "Insufficient material"
        else:
            value = 0.0
            outcome = "Move limit"
        
        if should_record:
            print(f"Game {game} ended by: {outcome}")
        
        # Update replay buffer for both sides by flipping the value each move
        for state, move in game_states:
            model.replay_buffer.push(
                state, 
                torch.zeros(get_move_space_size()).index_fill_(0, torch.tensor(model.move_to_index(move)), 1.0),
                torch.tensor([value])
            )
            value = -value  # Flip the value to reflect the opponent's perspective
        
        if game % 10 == 0:
            save_model_metric(game, model, model_filename)

def competitive_training(model, competitor, num_games=1000000, moves_per_game=10000, viz_every=1000):
    """
    Train the model by playing competitive matches against Stockfish.
    In this setup the training agent always plays as White (i.e. board.turn == True),
    while Stockfish plays as Black.
    """
    recorder = ChessVideoRecorder()
    
    # Use the updated wandb.run.name (random name with timestamp appended)
    model_filename = wandb.run.name
    
    for game in range(num_games):
        board = chess.Board()
        # Record only the states when the training model (White) makes a move.
        model_states = []
        should_record = args.record and (game % viz_every == 0)
        
        if should_record:
            recorder.start_game(game)
            recorder.save_frame(board)
        
        for move_num in range(moves_per_game):
            if board.is_game_over():
                break
            
            # Get current state (for potential training)
            state = board_to_tensor(board)
            
            if board.turn:  # Training agent's turn (White)
                temperature = max(1.0 - move_num / 30, 0.1)
                move = model.select_move(board, temperature)
                model_states.append((state, move))
            else:  # Competitor's turn (Black, Stockfish)
                move = competitor.select_move(board)
            
            board.push(move)
            
            if should_record:
                recorder.save_frame(board)
        
        if should_record:
            video_name = f"{model_filename}"
            recorder.end_game(video_name, framerate=2)
            print(f"\nVideo saved for game {game} as {video_name}!")
        
        # Determine game outcome from the training agent's perspective (playing as White)
        if board.is_checkmate():
            # If board.turn is True, then White (our agent) is to move and has been checkmated
            value = -1.0 if board.turn else 1.0
            outcome = "Checkmate"
        elif board.is_stalemate():
            value = 0.0
            outcome = "Stalemate"
        elif board.is_insufficient_material():
            value = 0.0
            outcome = "Insufficient material"
        else:
            value = 0.0
            outcome = "Move limit"
        
        if should_record:
            print(f"Game {game} ended by: {outcome}")
        
        # Update replay buffer for only the moves made by the training agent.
        # Since the model always plays as White, we do not flip the value.
        for state, move in model_states:
            model.replay_buffer.push(
                state,
                torch.zeros(get_move_space_size()).index_fill_(0, torch.tensor(model.move_to_index(move)), 1.0),
                torch.tensor([value])
            )
        
        # Training step
        if game % 10 == 0:
            save_model_metric(game, model, model_filename)

def evaluate_model(model, competitor, num_games=50):
    """Evaluate the model against a competitor agent in a series of games."""
    # Get computational metrics from the model.
    estimated_ram_mib, compressed_size_kib, cpu_freq = show_computational_metrics(model)
    wandb.log({"estimated_ram_mib": estimated_ram_mib, "compressed_size_kib": compressed_size_kib, "cpu_freq": cpu_freq.current})

    win_rate_against_best_model, draw_rate_against_best_model, loss_rate_against_best_model = run_evaluation(model, competitor, num_games=50)
    wandb.log({"win_rate_best_model": win_rate_against_best_model, "draw_rate_best_model": draw_rate_against_best_model, "loss_rate_best_model": loss_rate_against_best_model})

    # win_rate_against_stockfish, draw_rate_against_stockfish, loss_rate_against_stockfish = run_evaluation(model, ChessStockfishAgent(engine_path=stockfish_path, time_limit=0.1), num_games=50)
    # wandb.log({"win_rate_stockfish": win_rate_against_stockfish, "draw_rate_stockfish": draw_rate_against_stockfish, "loss_rate_stockfish": loss_rate_against_stockfish})
    
    sts_total, sts_percentage = evaluate_strategic_test_suite(model)
    wandb.log({"strategic_test_suite_total": sts_total, "strategic_test_suite_percentage": sts_percentage})
    
def save_model_metric(game, model, model_filename):
    loss = model.train_step()
    print(f"Game {game}, Loss: {loss}")
    gc.collect()
    
    # Log basic training metrics and checkpoint every 100 games.
    if game % 100 == 0:
        wandb.log({"loss": loss, "mode": args.mode, "game": game})
        model_path = f"{model_filename}.pth"
        model.save_model(model_path)
        wandb.save(model_path)
    
    # Every 1000 games, run a more expensive evaluation and log additional metrics.
    if game % 1000 == 0:
        # Evaluate the model against Stockfish by comparing with the best saved model.
        global BEST_MODEL_PATH
        best_model = ChessNeuralAgent()
        best_model.load_model(BEST_MODEL_PATH)
        best_model.eval()
        evaluate_model(model, best_model, num_games=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training for a chess agent in self-play or competitive mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["self-play", "competitive"],
        default="self-play",
        help="Training mode: 'self-play' for self-play reinforcement learning or 'competitive' for training against Stockfish."
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["neural", "mcts", "distill", "light", "giraffe"],
        default="neural",
        help="Specify the training agent type. Currently supported: 'neural', 'mcts', 'distill', 'light', 'giraffe'."
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["stockfish"],
        help="Specify the opponent for competitive mode. Only 'stockfish' is supported."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Additional notes to log in wandb."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000000,
        help="Number of games to play in training."
    )
    parser.add_argument(
    "--record",
    type=lambda x: x.lower() in ['true', '1', 'yes'],
    default=False,
    help="Enable video recording during training. Specify True or False (default: False)"
    )

    args = parser.parse_args()

    # Load configuration from config.json
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")


    # Extract configuration settings with fallbacks if necessary
    stockfish_path = config.get("engine", {}).get("stockfish_path", {})

    # Instantiate the training agent
    if args.agent == "neural":
        model = ChessNeuralAgent()
    elif args.agent == "mcts":
        model = ChessAlphaZeroAgent()
    elif args.agent == "distill":
        model = ChessDistillationAgent(stockfish_path=stockfish_path)
    elif args.agent == "light":
        model = ChessLightAgent()
    elif args.agent == "giraffe":
        model = GiraffeNeuralAgent()
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")
    
    # Initialize wandb without specifying a name so it generates one automatically.
    run = wandb.init(
        project="chess_rl",
        config=config,
        notes=args.notes
    )
    # Append current time to the randomly generated wandb run name.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run.name = f"{args.mode.capitalize()}_{timestamp}_{args.agent}_{run.name}"
    
    if args.mode == "self-play":
        self_play_training(model, num_games=args.num_games)
    elif args.mode == "competitive":
        if not args.opponent:
            raise ValueError("Competitive mode requires an opponent to be specified: 'stockfish'.")
        # Import the Stockfish agent
        competitor = ChessStockfishAgent(engine_path=stockfish_path, time_limit=0.1)
        competitive_training(model, competitor, num_games=args.num_games)