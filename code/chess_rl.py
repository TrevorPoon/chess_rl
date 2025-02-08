import chess
import torch
import torch.nn as nn
import torch.optim as optim
from utils.chess_recorder import ChessVideoRecorder
from utils.util import board_to_tensor, get_move_space_size
from agent_neural import ChessRL

def self_play_training(model, num_games=1000000, moves_per_game=1000, viz_every=50):
    """Train the model through self-play with video recording"""
    recorder = ChessVideoRecorder()
    
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
            
            # Select and make move
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
            recorder.end_game(framerate=2)
            print(f"\nVideo saved for game {game}!")
        
        # Game outcome
        if board.is_checkmate():
            value = 1.0 if board.turn else -1.0
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
        
        # Update replay buffer
        for state, move in game_states:
            model.replay_buffer.push(state, 
                                  torch.zeros(get_move_space_size()).index_fill_(0, 
                                      torch.tensor(model.move_to_index(move)), 1.0),
                                  torch.tensor([value]))
            value = -value  # Flip value for opponent's moves
        
        # Training step
        if game % 10 == 0:
            loss = model.train_step()
            print(f"Game {game}, Loss: {loss}")

            # Save the model after training
            model.save_model()

if __name__ == "__main__":
    chess_rl = ChessRL()
    self_play_training(chess_rl)