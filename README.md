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
Recommended Opening: R
Enter the current game state (e.g., RP, PS) where the first letter is your move and the second letter is the opponent's move.

--- Round 1 ---
Game State: RR
You - 0, Opponent - 0
Game state RR has counts [93, 80, 111]
Opponent will play S. Counter with R.

--- Round 2 ---
Game State: 
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
