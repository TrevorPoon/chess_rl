# Development of Resource-Constrained Chess Engines

This repository contains code for developing resource-constrained chess engines using reinforcement learning. You can train your model via self-play or by playing competitive matches against Stockfish.

## Prerequisites

- Python 3.13.1
- Required Python packages (see `install.sh`)
- [Stockfish](https://stockfishchess.org/download/) (download and install Stockfish from the official website)

> **Note:**  
> Make sure to update the path to the Stockfish executable in `config.json`.

## Running the Code

### Competitive Mode

To run competitive matches (where your neural network agent plays against Stockfish), execute:

```bash
python code/chess_rl.py --mode competitive --agent neural --opponent stockfish
```

### Self-Play Mode

In self-play mode, your neural network agent trains by playing against itself. To run self-play reinforcement learning, execute:

```bash
python code/chess_rl.py --mode self-play --agent neural
```