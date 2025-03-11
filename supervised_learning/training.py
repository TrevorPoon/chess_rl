import argparse
import os
import time
import pickle
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import wandb

# Import our PGN data loader and mapping functions.
from utils.load_pgn import PGNIterableDataset, build_move_mapping
# Import evaluation functions.
from eval import evaluate_random_model, evaluate_strategic_test_suite

def train_model(args):
    # Initialize wandb with auto-generated run name and additional config.
    run = wandb.init(
        project="chess_sl",
        notes=args.notes,
        config={
            "agent": args.agent,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "limit_files": args.limit_files
        }
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run.name = f"{args.mode.capitalize()}_{timestamp}_{args.agent}_{run.name}"
    wandb.config.update({"run_name": run.name})

    # Dynamically import the specified agent module.
    agent_module = importlib.import_module(f"agent.{args.agent}")
    
    pgn_dir = os.path.join("data", "train")
    
    # Build the move mapping from a limited set of PGN files.
    move_to_int = build_move_mapping(pgn_dir, limit_files=args.limit_files)
    num_classes = len(move_to_int)
    print(f"Unique moves: {num_classes}")
    
    # Save move mapping for later use.
    os.makedirs("model", exist_ok=True)
    pickle_path = os.path.join("model", "move_to_int.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(move_to_int, file)
    
    # Create an iterable dataset from PGN files.
    dataset = PGNIterableDataset(pgn_dir, move_to_int, limit_files=args.limit_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The agent module is expected to define a class called 'Agent'.
    model = agent_module.Agent(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Prepare a CSV file to log evaluation metrics.
    csv_filename = f"data/result/{run.name}_evaluation_results.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Loss", "Wins_Against_Random", "Draws_Against_Random", "Losses_Against_Random", "STS Score", "STS Percentage"])
    
    # Training loop.
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_counter = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            batch_counter += 1
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / batch_counter
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec")
        
        # Convert mapping for evaluation.
        int_to_move = {v: k for k, v in move_to_int.items()}
        wins_agaisnt_random, draws_against_random, losses_against_random = evaluate_random_model(model, device, int_to_move, num_games=100)
        sts_score, sts_percentage = evaluate_strategic_test_suite(model, device, int_to_move)
        
        print(f"Evaluation against Random Agent after Epoch {epoch+1}: Wins: {wins_agaisnt_random}, Draws: {draws_against_random}, Losses: {losses_against_random}")
        print(f"STS Score: {sts_score}, STS Percentage: {sts_percentage}")
        
        # Log statistics to wandb.
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "epoch_time": epoch_time,
            "wins_agaisnt_random": wins_agaisnt_random,
            "draws_against_random": draws_against_random,
            "losses_against_random": losses_against_random,
            "sts_score": sts_score,
            "sts_percentage": sts_percentage
        })
        
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch+1, avg_loss, wins_agaisnt_random, draws_against_random, losses_against_random, sts_score, sts_percentage])
        
        # Save the model checkpoint with run name.
        model_path = os.path.join("model", f"{run.name}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
    
    print("Training complete. Model and mapping saved.")
    run.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a chess model.")

    parser.add_argument('--agent', type=str, default="agent_basic",
                        help="Module name of the agent to use (e.g. agent_basic)")
    parser.add_argument('--limit_files', type=int, default=100,
                        help="Limit the number of PGN files to process")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--notes', type=str, default="",
                        help="Additional notes for wandb run")
    parser.add_argument('--mode', type=str, default="train",
                        help="Mode of operation: train or eval")
    args = parser.parse_args()

    train_model(args)
