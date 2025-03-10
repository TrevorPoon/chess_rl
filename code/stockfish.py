#!/usr/bin/env python3
import chess
import json
import os
from built_in_agent.agent_stockfish import ChessStockfishAgent

def load_stockfish_path():
    """
    Attempt to load the Stockfish engine path from a config.json file.
    If not found or misconfigured, default to "stockfish".
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            # Expecting the JSON structure to contain "engine": { "stockfish_path": <path> }
            return config.get("engine", {}).get("stockfish_path", "stockfish")
        except Exception as e:
            print(f"Error loading config: {e}. Using default engine path.")
    return "stockfish"

def play_game(stockfish_path):
    """
    Plays a single game between two Stockfish agents.
    If the game exceeds 500 moves, it will be terminated and declared a draw.
    Returns the game result as a string: "1-0", "0-1", or "1/2-1/2".
    """
    board = chess.Board()
    agent_white = ChessStockfishAgent(engine_path=stockfish_path, time_limit=0.1)
    agent_black = ChessStockfishAgent(engine_path=stockfish_path, time_limit=0.1)
    
    move_count = 0
    while not board.is_game_over() and move_count < 500:
        if board.turn:  # White to move
            move, _ = agent_white.select_move(board)
        else:          # Black to move
            move, _ = agent_black.select_move(board)
        board.push(move)
        move_count += 1

    # If the game didn't naturally finish and move limit was reached, declare a draw.
    if move_count >= 500 and not board.is_game_over():
        print("Move limit reached (500 moves). Ending game as a draw.")
        return "1/2-1/2"
    else:
        return board.result()

def main():
    stockfish_path = load_stockfish_path()
    print(f"Using Stockfish engine from: {stockfish_path}")
    
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    num_games = 100

    for i in range(num_games):
        result = play_game(stockfish_path)
        results[result] += 1
        print(f"Game {i+1}: Result = {result}")
    
    print("\nFinal Results after 100 games:")
    print(f"  White wins (1-0): {results['1-0']}")
    print(f"  Black wins (0-1): {results['0-1']}")
    print(f"  Draws (1/2-1/2): {results['1/2-1/2']}")

if __name__ == "__main__":
    main()
