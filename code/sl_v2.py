#!/usr/bin/env python
import argparse
import os
import json
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader

# Import the utility to get move space size
from utils.util import get_move_space_size

# Import your agent implementations.
from rl_agent.agent_neural import ChessNeuralAgent
from eval import run_evaluation, evaluate_strategic_test_suite
from built_in_agent.agent_random import ChessRandomAgent
# (Other agents are available if needed)

class PGNDataset(IterableDataset):
    """
    A lazy-loading dataset that streams training examples from a PGN file.
    Each example is a tuple: (state_vector, one-hot policy target tensor, value target).
    A fixed random seed (and ratio) is used to split examples into training and validation sets.
    """
    def __init__(self, model, pgn_file, mode='train', train_ratio=0.9, seed=42):
        self.model = model
        self.pgn_file = pgn_file
        self.mode = mode  # 'train' or 'val'
        self.train_ratio = train_ratio
        self.seed = seed

    def __iter__(self):
        # Initialize a random generator with a fixed seed to ensure a consistent split.
        rng = np.random.RandomState(self.seed)
        with open(self.pgn_file, errors='ignore') as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                # Only process decisive games.
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    white_value = 1.0
                    black_value = -1.0
                elif result == "0-1":
                    white_value = -1.0
                    black_value = 1.0
                else:
                    continue  # skip drawn or non-decisive games

                board = game.board()
                for move in game.mainline_moves():
                    # Compute the state vector and targets.
                    state = self.model.board_to_vector(board)  # Expected shape: (channels, 8, 8)
                    move_index = self.model.move_to_index(move)
                    policy_target = torch.zeros(get_move_space_size(), dtype=torch.float32)
                    policy_target[move_index] = 1.0
                    if board.turn:  # White to move.
                        value_target = white_value
                    else:
                        value_target = black_value
                    board.push(move)
                    
                    # Use the RNG to decide whether this example is for training or validation.
                    if rng.rand() < self.train_ratio:
                        if self.mode == 'train':
                            yield (state, policy_target, value_target)
                    else:
                        if self.mode == 'val':
                            yield (state, policy_target, value_target)

def supervised_training(model, train_dataset, val_dataset, epochs=3, batch_size=128):
    """
    Performs supervised training using lazy-loaded data from the PGNDataset.
    The examples are streamed from disk (no full in-memory conversion).
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Starting supervised training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for state_batch, policy_batch, value_batch in train_loader:
            model.optimizer.zero_grad()
            # Compute loss. Assumes state_batch, policy_batch, and value_batch are already tensors.
            loss_val = model.minimise_loss(state_batch, policy_batch, value_batch, batch_size)
            loss_val.backward()
            model.optimizer.step()

            total_loss += loss_val.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Training Average Loss: {avg_train_loss}")
        wandb.log({"sl_epoch": epoch+1, "sl_train_loss": avg_train_loss}, step=epoch+1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for state_batch, policy_batch, value_batch in val_loader:
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

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Learning for Chess Agent using PGN data with Data Augmentation"
    )
    parser.add_argument("--pgn_file", type=str, default="data/ficsgamesdb_2024_standard2000_nomovetimes_14726.pgn",
                        help="Path to the PGN file containing game data from FICS")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--agent", type=str, choices=["neural", "mcts", "distill", "light", "giraffe","nnue"],
                        default="giraffe", help="Specify the training agent type")
    parser.add_argument("--notes", type=str, default="",
                        help="Additional notes to log in wandb")
    args = parser.parse_args()

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
    elif args.agent == "light":
        from rl_agent.agent_light import ChessLightAgent
        model = ChessLightAgent()
    elif args.agent == "giraffe":
        from rl_agent.agent_giraffe import GiraffeChessAgent
        model = GiraffeChessAgent()
    elif args.agent == "nnue":
        from rl_agent.agent_NNUE import NNUEChessAgent
        model = NNUEChessAgent()
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    # Create lazy-loaded training and validation datasets.
    train_dataset = PGNDataset(model, args.pgn_file, mode='train', train_ratio=0.9, seed=42)
    val_dataset = PGNDataset(model, args.pgn_file, mode='val', train_ratio=0.9, seed=42)
    
    print("Starting training using lazy-loading PGN dataset...")
    supervised_training(model, train_dataset, val_dataset, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
