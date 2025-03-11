import time
import chess
from typing import List, Dict

def read_epd_file(epd_path: str) -> List[Dict[str, object]]:
    """
    Reads an EPD file and returns test cases with FEN and expected moves.
    """
    def parse_epd_line(line: str) -> Dict[str, object]:
        parts = line.split(';')
        if not parts:
            return {}
        fen_tokens = parts[0].split()
        if len(fen_tokens) < 4:
            return {}
        fen = " ".join(fen_tokens[:4]) + " 0 1"
        
        expected_moves = {}
        notation_moves = []
        scores = []
        for part in parts[1:]:
            part = part.strip()
            if part.startswith("c9"):
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
            if not line or line.startswith("#"):
                continue
            case = parse_epd_line(line)
            if case:
                test_cases.append(case)
    return test_cases

def evaluate_strategic_test_suite(model, device, int_to_move, epd_file_path="data/STS1-STS15_LAN_v3.epd"):
    """
    Evaluates the model on a strategic test suite loaded from an EPD file.
    """
    test_cases = read_epd_file(epd_file_path)
    total_score = 0
    max_score = len(test_cases) * 10
    print(f"Starting STS evaluation on {len(test_cases)} positions")
    start_time = time.perf_counter()
    
    for idx, test in enumerate(test_cases, start=1):
        fen = test.get("fen")
        expected_moves = test.get("expected_moves", {})
        board = chess.Board(fen)
        move_uci = model.predict_move(board, device, int_to_move)
        score_awarded = expected_moves.get(move_uci, 0)
        total_score += score_awarded
    
    overall_elapsed = time.perf_counter() - start_time
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    print(f"STS Evaluation: Total score {total_score}/{max_score} ({percentage:.2f}%) over {overall_elapsed:.2f} sec")
    return total_score, percentage

def evaluate_random_model(model, device, int_to_move, num_games=100):
    """
    Evaluates the model by playing num_games against a random agent.
    """
    import random
    wins = 0
    draws = 0
    losses = 0
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:
                move_uci = model.predict_move(board, device, int_to_move)
                if move_uci is None:
                    move = random.choice(list(board.legal_moves))
                else:
                    move = chess.Move.from_uci(move_uci)
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)
        result = board.result()
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
    return wins, draws, losses
