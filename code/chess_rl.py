import argparse
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import wandb
from datetime import datetime

from utils.chess_recorder import ChessVideoRecorder
from utils.util import *
from rl_agent.agent_neural import ChessNeuralAgent

def self_play_training(model, model_type, num_games=1000000, moves_per_game=1000, viz_every=50):
    """Train the model through self-play with video recording."""
    recorder = ChessVideoRecorder()

    # Use the updated wandb.run.name (random name with timestamp appended)
    model_filename = wandb.run.name
    
    for game in range(num_games):
        board = chess.Board()
        game_states = []
        should_record = game % viz_every == 0
        
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
        
        # Training step
        if game % 10 == 0:
            loss = model.train_step()
            print(f"Game {game}, Loss: {loss}")
            wandb.log({"game": game, "loss": loss, "mode": "self-play"})
            
            # Save the model periodically
            if game % 100 == 0:
                model_path = f"{model_filename}.pth"
                model.save_model(model_path)
                wandb.save(model_path)

def competitive_training(model, model_type, competitor, num_games=1000000, moves_per_game=1000, viz_every=50):
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
        should_record = game % viz_every == 0
        
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
            loss = model.train_step()
            print(f"Game {game}, Loss: {loss}")
            wandb.log({"game": game, "loss": loss, "mode": "competitive"})
            
            if game % 100 == 0:
                model_path = f"{model_filename}.pth"
                model.save_model(model_path)
                wandb.save(model_path)

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
        choices=["neural"],
        default="neural",
        help="Specify the training agent type. Currently supported: 'neural'."
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

    # Initialize wandb without specifying a name so it generates one automatically.
    run = wandb.init(
        project="chess_rl",
        config=config,
        notes=args.notes
    )
    # Append current time to the randomly generated wandb run name.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run.name = f"{args.mode.capitalize()}_{args.agent}_{run.name}_{timestamp}"

    # Instantiate the training agent (currently only 'neural' is supported)
    if args.agent == "neural":
        model = ChessNeuralAgent()
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")
    
    if args.mode == "self-play":
        self_play_training(model, args.agent)
    elif args.mode == "competitive":
        if not args.opponent:
            raise ValueError("Competitive mode requires an opponent to be specified: 'stockfish'.")
        # Import the Stockfish agent
        from built_in_agent.agent_stockfish import ChessStockfishAgent
        competitor = ChessStockfishAgent(engine_path=stockfish_path, time_limit=0.1)
        competitive_training(model, args.agent, competitor)