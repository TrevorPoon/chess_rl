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
from sklearn.model_selection import train_test_split

# Import the utility to get move space size
from utils.util import get_move_space_size

# Import your agent implementations.
from rl_agent.agent_neural import ChessNeuralAgent
from eval import run_evaluation, evaluate_strategic_test_suite
from built_in_agent.agent_random import ChessRandomAgent
# (Other agents are available if needed)


#############################
# PGN Reading & Processing  #
#############################

def get_games_from_file(filename):
    """
    Reads a PGN file and returns a list of chess.pgn.Game objects.
    """
    games = []
    # Use a context manager for safe file handling.
    with open(filename, errors='ignore') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def process_game(game, model):
    """
    Processes one PGN game into a list of training examples.
    Each example is a tuple: (state_vector, one-hot policy target tensor, value target).
    The value target is set from the perspective of the player who moved.
    Drawn or non-decisive games are skipped.
    """
    dataset = []
    result = game.headers.get("Result", "*")
    
    # Only process decisive games.
    if result == "1-0":
        white_value = 1.0
        black_value = -1.0
    elif result == "0-1":
        white_value = -1.0
        black_value = 1.0
    else:
        return dataset

    board = game.board()
    for move in game.mainline_moves():
        # Original example:
        state = model.board_to_vector(board)  # Expected shape: (channels, 8, 8)
        move_index = model.move_to_index(move)
        policy_target = torch.zeros(get_move_space_size(), dtype=torch.float32)
        policy_target[move_index] = 1.0
        if board.turn:  # White to move.
            value_target = white_value
        else:
            value_target = black_value
        dataset.append((state, policy_target, value_target))
        
        board.push(move)

    return dataset

def build_dataset_from_pgn(model, pgn_file="data/ficsgamesdb_2024_standard2000_nomovetimes_14726.pgn"):
    """
    Reads all games from the given PGN file and processes them into training examples.
    Returns a tuple (train_dataset, val_dataset) after splitting.
    """
    games = get_games_from_file(pgn_file)
    full_dataset = []
    for game in games:
        game_data = process_game(game, model)
        full_dataset.extend(game_data)
    
    print(f"Total examples before split: {len(full_dataset)}")
    # Split the dataset into 90% training and 10% validation
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.10, random_state=42)
    print(f"Training examples: {len(train_dataset)} | Validation examples: {len(val_dataset)}")
    return train_dataset, val_dataset

#############################
# Supervised Training Code  #
#############################

def supervised_training(model, train_dataset, val_dataset, epochs=3, batch_size=128):
    """
    Performs supervised training on the provided training dataset and evaluates on validation data.
    The dataset is a list of tuples: (state, policy target tensor, value target).
    """
    # Prepare training tensors.
    train_states = torch.tensor(np.array([x[0] for x in train_dataset], dtype=np.float32))
    train_policy = torch.stack([x[1] for x in train_dataset])
    train_value = torch.tensor(np.array([x[2] for x in train_dataset], dtype=np.float32)).view(-1, 1)

    train_ds = torch.utils.data.TensorDataset(train_states, train_policy, train_value)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Prepare validation tensors.
    val_states = torch.tensor(np.array([x[0] for x in val_dataset], dtype=np.float32))
    val_policy = torch.stack([x[1] for x in val_dataset])
    val_value = torch.tensor(np.array([x[2] for x in val_dataset], dtype=np.float32)).view(-1, 1)

    val_ds = torch.utils.data.TensorDataset(val_states, val_policy, val_value)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Starting supervised training for {epochs} epochs on {len(train_ds)} training examples...")
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        for state_batch, policy_batch, value_batch in train_loader:
            model.optimizer.zero_grad()
            state_batch = state_batch.to(model.device)
            policy_batch = policy_batch.to(model.device)
            value_batch = value_batch.to(model.device)

            loss_val = model.minimise_loss(state_batch, policy_batch, value_batch, batch_size)
            loss_val.backward()
            model.optimizer.step()
            
            total_loss += loss_val.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Training Average Loss: {avg_train_loss}")
        wandb.log({"sl_epoch": epoch+1, "sl_train_loss": avg_train_loss}, step=epoch+1)

        # Evaluate on validation set.
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for state_batch, policy_batch, value_batch in val_loader:
                state_batch = state_batch.to(model.device)
                policy_batch = policy_batch.to(model.device)
                value_batch = value_batch.to(model.device)
                loss_val = model.minimise_loss(state_batch, policy_batch, value_batch, batch_size)
                val_loss += loss_val.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Validation Average Loss: {avg_val_loss}")
        wandb.log({"sl_val_loss": avg_val_loss}, step=epoch+1)

        # Run evaluation against a random agent.
        sl_win_rate, sl_draw_rate, sl_loss_rate = run_evaluation(model, ChessRandomAgent(), num_games=50)
        wandb.log({
            "sl_win_rate_against_random": sl_win_rate,
            "sl_draw_rate_against_random": sl_draw_rate,
            "sl_loss_rate_against_random": sl_loss_rate
        }, step=epoch+1)
    
        # Evaluate on strategic test suite.
        sts_total, sts_percentage = evaluate_strategic_test_suite(model)
        wandb.log({"sl_strategic_test_suite_total": sts_total, "sl_strategic_test_suite_percentage": sts_percentage}, step=epoch+1)
    
    # Optionally, save the model after training.
    # model_filename = f"sl_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    # model.save_model(model_filename)
    # print(f"Supervised pre-training complete. Model saved as {model_filename}")

#############################
# Main Function             #
#############################

def main():
    parser = argparse.ArgumentParser(description="Supervised Learning for Chess Agent using PGN data with Data Augmentation")
    parser.add_argument("--pgn_file", type=str, default="data/ficsgamesdb_2024_standard2000_nomovetimes_14726.pgn",
                        help="Path to the PGN file containing game data from FICS")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--agent", type=str, choices=["neural", "mcts", "distill", "light", "giraffe"],
                        default="giraffe", help="Specify the training agent type")
    parser.add_argument("--notes", type=str, default="",
                        help="Additional notes to log in wandb")
    args = parser.parse_args()
    
    # Load configuration from config.json (assumed to be in the parent directory)
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
    train_dataset, val_dataset = build_dataset_from_pgn(model, args.pgn_file)
    print(f"Total training examples (after filtering): {len(train_dataset) + len(val_dataset)}")
    
    # Run supervised training for the specified number of epochs.
    supervised_training(model, train_dataset, val_dataset, args.epochs, args.batch_size)
    
if __name__ == "__main__":
    main()
