# Adaptive RPS Markov AI

An adaptive Rock-Paper-Scissors AI built in Python using variable-order Markov chains to model and predict opponent behaviour from historical move patterns.

## Features
- Variable-order Markov chain prediction
- Probabilistic opponent move modelling
- Adaptive prediction based on historical gameplay
- Continual self-adjustment / fine-tuning support
- Interactive player-vs-bot gameplay
- Persistent game result logging

## Technologies
- Python
- JSON
- Markov Chains
- Probabilistic Modelling

## Overview
The system trains an n-th order Markov model on historical Rock-Paper-Scissors game sequences. During gameplay, the AI analyses recent move patterns to estimate the opponent’s most likely next action and recommends the optimal counter move.

The project explores probabilistic prediction systems, sequential pattern analysis, and adaptive behavioural modelling through a simple competitive environment.

## Example Gameplay
```txt
Let's play Rock-Paper-Scissors!

--- Round 1 ---
Game State: RP
You - 1, Opponent - 0

Game state PR has counts [5, 12, 3]
Opponent will play P. Counter with S.
```

## Files
- `markov.py` — training, prediction, fine-tuning, and gameplay logic
- `RPS.txt` — historical training data
- `results.txt` — stored gameplay results and outcomes

## Run
```bash
python3 markov.py
```

## Future Improvements
- GUI-based gameplay interface
- Dynamic order selection
- Statistical accuracy analysis
- Visualization of transition probabilities
- Expanded adaptive learning systems
