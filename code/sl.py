#!/usr/bin/env python
import argparse
import os
import json
import re
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from datetime import datetime

# Import the utility to get move space size
from utils.util import get_move_space_size

# Import your agent implementations.
from rl_agent.agent_neural import ChessNeuralAgent
# (Other agents are available if needed)

#############################
# PGN Reading & Processing  #
#############################

# Copy the PGN reading function from your FICS reference.
def get_games_from_file(filename):
    """
    Reads a PGN file and returns a list of chess.pgn.Game objects.
    """
    pgn = open(filename, errors='ignore')
    offsets = []
    while True:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break
        offsets.append(offset)
    games = []
    for offset in offsets:
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        games.append(game)
    return games

def process_game(game, model):
    """
    Processes one PGN game into a list of training examples.
    Each example is a tuple:
      (state_vector, one-hot policy target, value target)
    The value target is set from the perspective of the player who moved.
    """
    dataset = []
    result = game.headers.get("Result", "*")
    # Set outcome values: for white moves, value=1 if white wins, -1 if loses, 0 for draw.
    if result == "1-0":
        white_value = 1.0
        black_value = -1.0
    elif result == "0-1":
        white_value = -1.0
        black_value = 1.0
    else:
        white_value = 0.0
        black_value = 0.0

    board = game.board()
    for move in game.mainline_moves():
        # Get the current board representation as a vector.
        state = model.board_to_vector(board)
        # Convert the move to an index, then create a one-hot vector target.
        move_index = model.move_to_index(move)
        policy_target = torch.zeros(get_move_space_size(), dtype=torch.float32)
        policy_target[move_index] = 1.0

        # Assign the value target based on whose turn it is.
        if board.turn:  # White to move.
            value_target = white_value
        else:
            value_target = black_value

        dataset.append((state, policy_target.numpy(), value_target))
        board.push(move)
    return dataset

def build_dataset_from_pgn(model, pgn_file="data/ficsgamesdb_201801_standard2000_nomovetimes_14314.pgn"):
    """
    Reads all games from the given PGN file and processes them into training examples.
    """
    games = get_games_from_file(pgn_file)
    dataset = []
    for game in games:
        game_data = process_game(game, model)
        dataset.extend(game_data)
    return dataset

#############################
# Supervised Training Code  #
#############################

def supervised_training(model, dataset, epochs, batch_size):
    """
    Performs supervised training on the provided dataset.
    The dataset is a list of tuples: (state, policy target, value target).
    """
    # Convert dataset into numpy arrays.
    states = np.array([x[0] for x in dataset], dtype=np.float32)
    policy_targets = np.array([x[1] for x in dataset], dtype=np.float32)
    value_targets = np.array([x[2] for x in dataset], dtype=np.float32).reshape(-1, 1)

    # Create torch tensors and a DataLoader.
    tensor_states = torch.tensor(states)
    tensor_policy = torch.tensor(policy_targets)
    tensor_value = torch.tensor(value_targets)
    ds = torch.utils.data.TensorDataset(tensor_states, tensor_policy, tensor_value)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"Starting supervised training for {epochs} epochs on {len(ds)} examples...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for state_batch, policy_batch, value_batch in loader:
            model.optimizer.zero_grad()
            state_batch = state_batch.to(model.device)
            policy_batch = policy_batch.to(model.device)
            value_batch = value_batch.to(model.device)
            # Forward pass through the model.
            pred_policy, pred_value = model.model(state_batch)
            # If the predicted policy vector is larger than our target, pad the target.
            if pred_policy.size(1) > policy_batch.size(1):
                padding = torch.zeros(policy_batch.size(0), pred_policy.size(1) - policy_batch.size(1), device=model.device)
                policy_batch = torch.cat([policy_batch, padding], dim=1)
            # Compute cross-entropy loss for the policy head.
            loss_policy = -torch.sum(policy_batch * torch.log(pred_policy + 1e-8)) / state_batch.size(0)
            # Compute mean squared error loss for the value head.
            loss_value = torch.mean((value_batch.squeeze() - pred_value.squeeze())**2)
            loss = loss_policy + loss_value
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        wandb.log({"sl_epoch": epoch+1, "sl_loss": avg_loss})
    
    # Save the model after supervised pre-training.
    # model_filename = f"sl_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    # model.save_model(model_filename)
    # print(f"Supervised pre-training complete. Model saved as {model_filename}")

#############################
# Main Function             #
#############################

def main():
    parser = argparse.ArgumentParser(description="Supervised Learning for Chess Agent using PGN data")
    parser.add_argument("--pgn_file", type=str, default="data/ficsgamesdb_201801_standard2000_nomovetimes_14314.pgn",
                        help="Path to the PGN file containing game data from FICS")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--agent", type=str, choices=["neural", "mcts", "distill", "light", "giraffe"],
                        default="giraffe", help="Specify the training agent type")
    parser.add_argument("--notes", type=str, default="",
                        help="Additional notes to log in wandb")
    args = parser.parse_args()
    
    # Load configuration from config.json (assumed to be in the same directory as sl.py)
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    # Initialize wandb for experiment tracking.
    run = wandb.init(
        project="chess_rl",
        notes=args.notes,
        config={"agent": args.agent, "pgn_file": args.pgn_file}
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run.name = f"Supervised_{timestamp}_{args.agent}_{run.name}"
    
    # Instantiate the agent based on the selected type.
    if args.agent == "neural":
        model = ChessNeuralAgent()
    elif args.agent == "mcts":
        from rl_agent.agent_mcts import ChessAlphaZeroAgent
        model = ChessAlphaZeroAgent()
    elif args.agent == "distill":
        from rl_agent.agent_distill import ChessDistillationAgent
        stockfish_path = config.get("engine", {}).get("stockfish_path", "")
        model = ChessDistillationAgent(stockfish_path=stockfish_path)
    elif args.agent == "light":
        from rl_agent.agent_light import ChessLightAgent
        model = ChessLightAgent()
    elif args.agent == "giraffe":
        from rl_agent.agent_giraffe import GiraffeChessAgent
        model = GiraffeChessAgent()
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")
    
    # Build the supervised learning dataset from the PGN file.
    print("Reading PGN file and building the training dataset...")
    dataset = build_dataset_from_pgn(model, args.pgn_file)
    print(f"Total training examples: {len(dataset)}")
    
    # Run supervised training for the specified number of epochs.
    supervised_training(model, dataset, args.epochs, args.batch_size)
    
if __name__ == "__main__":
    main()
