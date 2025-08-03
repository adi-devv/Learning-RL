# Snakeeyy: A Turtle-Based Snake Game with Pathfinding and Q-Learning

## Overview
Snakeeyy is a Python-based Snake game built with the `turtle` module. It features a grid-based environment (20x20 grid) with wrap-around boundaries and an AI-driven snake using **breadth-first search (BFS)** pathfinding and **Q-learning elements** to navigate toward food while avoiding collisions. The game includes dynamic speed control, score tracking, and a reward system to guide the snake’s behavior.

## Features
- **Gameplay**: Classic Snake game where the snake grows by eating food (light coral circles) and avoids self-collision or excessive steps (max 2000 steps per episode).
- **AI**: Combines BFS for safe pathfinding to food with a partial Q-learning approach for action selection (up, left, down, right).
- **Controls**:
  - **Up/Down arrows**: Adjust movement speed (0.01–0.2 s/step).
  - **Space**: Pause/resume the game.
- **Rewards**: Food collection (+20), wall crossings (+0.3), high score (+50), step penalty (-0.1), turn penalty (-0.1), collision penalty (-1000).
- **Score System**: Tracks current and high scores, saved to `Highscore.txt`.

## Technologies
- **Python Libraries**:
  - `turtle`: Renders the 700x750 game interface with a light green background.
  - `time`: Controls game speed via delays.
  - `random`: Generates random food positions, avoiding snake overlap.
  - `math`: Computes distances for collision detection.
  - `collections.deque`: Implements BFS for pathfinding.
- **AI Components**:
  - **BFS**: Finds collision-free paths to food (`find_safest_path`).
  - **Q-Learning**: Partial implementation with a Q-table, learning rate (`alpha=0.2`), discount factor (`gamma=0.9`), and exploration rate (`epsilon=0.3` with decay), but lacks Q-table updates.

## Installation
1. **Prerequisites**: Python 3.6+ with standard libraries (`turtle`, `time`, `random`, `math`, `collections`).
2. Clone or download the repository:
   ```bash
   git clone https://github.com/adi-devv/Snake-Game-RL.git
   cd Snake-Game-RL
