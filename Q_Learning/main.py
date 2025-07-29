import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import random
from collections import deque

# Initialize Pygame with better settings

pygame.init()
width, height = 800, 800  # Larger window for better visualization
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("RL Maze Solver with Enhanced Visualization")
cell_size = width // 10  # For 10x10 maze
clock = pygame.time.Clock()  # For controlling frame rate

# Constants
MAZE_SIZE = 10
WALL_DENSITY = 0.25  # 25% walls
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 200  # Prevent infinite episodes

# Colors with better visual distinction
COLORS = {
    'background': (240, 240, 240),
    'wall': (50, 50, 50),
    'start': (30, 144, 255),  # Dodger Blue
    'goal': (50, 205, 50),  # Lime Green
    'agent': (220, 20, 60),  # Crimson Red
    'path': (255, 165, 0),  # Orange
    'visited': (173, 216, 230),  # Light Blue
    'text': (0, 0, 0),
    'grid': (200, 200, 200)
}


# Enhanced maze generation with guaranteed solvability
def generate_maze():
    maze = np.zeros((MAZE_SIZE, MAZE_SIZE), dtype=int)
    maze[0, 0] = 2  # Start
    maze[-1, -1] = 3  # Goal

    # Place walls with controlled density
    wall_count = 0
    max_walls = int(MAZE_SIZE * MAZE_SIZE * WALL_DENSITY)

    while wall_count < max_walls:
        i, j = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
        if maze[i, j] == 0 and (i, j) not in [(0, 0), (MAZE_SIZE - 1, MAZE_SIZE - 1)]:
            maze[i, j] = 1
            wall_count += 1

    # Ensure path exists using BFS
    def is_solvable():
        visited = set()
        queue = deque([(0, 0)])
        while queue:
            i, j = queue.popleft()
            if (i, j) == (MAZE_SIZE - 1, MAZE_SIZE - 1):
                return True
            if (i, j) in visited:
                continue
            visited.add((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < MAZE_SIZE and 0 <= nj < MAZE_SIZE and maze[ni, nj] != 1:
                    queue.append((ni, nj))
        return False

    # Remove walls until maze is solvable
    while not is_solvable():
        i, j = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
        if maze[i, j] == 1:
            maze[i, j] = 0

    return maze


maze = generate_maze()

# Improved Q-table initialization with better scaling
q_table = np.random.uniform(low=-0.1, high=0.1, size=(MAZE_SIZE, MAZE_SIZE, 4))

# Hyperparameters with better tuning
alpha = 0.2  # Higher learning rate for faster convergence
gamma = 0.95  # Slightly higher discount factor
epsilon = 1.0  # Start with full exploration
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Slower decay for better exploration

rewards = []  # Track rewards for plotting
success_rate = []  # Track success rate over time
steps_per_episode = []  # Track efficiency improvement


# Enhanced visualization function
def draw_maze(agent_pos, episode, path, visited=None, is_final=False, fps=60):
    screen.fill(COLORS['background'])

    # Draw visited cells (except in final visualization)
    if visited and not is_final:
        for i, j in visited:
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, COLORS['visited'], rect)

    # Draw maze elements
    for i in range(MAZE_SIZE):
        for j in range(MAZE_SIZE):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            if maze[i, j] == 1:
                pygame.draw.rect(screen, COLORS['wall'], rect)
            elif maze[i, j] == 2:
                pygame.draw.rect(screen, COLORS['start'], rect)
            elif maze[i, j] == 3:
                pygame.draw.rect(screen, COLORS['goal'], rect)
            pygame.draw.rect(screen, COLORS['grid'], rect, 1)  # Grid lines

    # Draw path with gradient from start to current position
    if len(path) >= 2:
        for k in range(1, len(path)):
            i_prev, j_prev = path[k - 1]
            i_curr, j_curr = path[k]
            # Calculate gradient color based on path position
            color_factor = k / len(path)
            path_color = (
                int(COLORS['path'][0] * (1 - color_factor) + COLORS['agent'][0] * color_factor),
                int(COLORS['path'][1] * (1 - color_factor) + COLORS['agent'][1] * color_factor),
                int(COLORS['path'][2] * (1 - color_factor) + COLORS['agent'][2] * color_factor)
            )
            start_pos = (j_prev * cell_size + cell_size // 2, i_prev * cell_size + cell_size // 2)
            end_pos = (j_curr * cell_size + cell_size // 2, i_curr * cell_size + cell_size // 2)
            pygame.draw.line(screen, path_color, start_pos, end_pos, 4)

    # Draw agent
    agent_rect = pygame.Rect(
        agent_pos[1] * cell_size + cell_size // 4,
        agent_pos[0] * cell_size + cell_size // 4,
        cell_size // 2,
        cell_size // 2
    )
    pygame.draw.ellipse(screen, COLORS['agent'], agent_rect)

    # Draw information overlay
    font = pygame.font.SysFont('Arial', 24)
    episode_text = f'Episode: {episode}' if not is_final else 'Final Path'
    text_surface = font.render(episode_text, True, COLORS['text'])
    screen.blit(text_surface, (10, 10))

    # Additional info during training
    if not is_final:
        epsilon_text = f'ε: {epsilon:.3f}'
        epsilon_surface = font.render(epsilon_text, True, COLORS['text'])
        screen.blit(epsilon_surface, (10, 40))

    pygame.display.flip()
    clock.tick(fps)  # Control frame rate


# Training with enhanced reward system and visualization
start_time = time.time()
print(f"Training started at: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

for episode in range(EPISODES):
    state = (0, 0)  # Start position
    path = [state]
    visited = set([state])
    total_reward = 0
    done = False
    success = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        steps += 1

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit

        # Calculate next state with boundary checks
        next_state = list(state)
        if action == 0:  # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Right
            next_state[1] = min(MAZE_SIZE - 1, state[1] + 1)
        elif action == 2:  # Down
            next_state[0] = max(0, min(MAZE_SIZE - 1, state[0] + 1))
        elif action == 3:  # Left
            next_state[1] = max(0, state[1] - 1)
        next_state = tuple(next_state)

        # Enhanced reward system
        if maze[next_state] == 1:  # Hit wall
            reward = -1
            next_state = state  # Stay in current state
        elif maze[next_state] == 3:  # Reached goal
            reward = 10
            done = True
            success = True
        else:
            # Reward based on progress toward goal (Manhattan distance)
            current_dist = (MAZE_SIZE - 1 - state[0]) + (MAZE_SIZE - 1 - state[1])
            next_dist = (MAZE_SIZE - 1 - next_state[0]) + (MAZE_SIZE - 1 - next_state[1])
            reward = 0.5 if next_dist < current_dist else -0.2

            # Small penalty for revisiting cells
            if next_state in visited:
                reward -= 0.3

        total_reward += reward
        visited.add(next_state)

        # Q-learning update with clipping
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - q_table[state[0], state[1], action]
        q_table[state[0], state[1], action] += alpha * td_error

        # Update state and path
        if next_state != state:
            state = next_state
            path.append(state)

        # Visualize every 10th episode or last 10 episodes for performance
        if episode % 10 == 0 or episode >= EPISODES - 10:
            draw_maze(state, episode, path, visited)

    # Update epsilon with minimum threshold
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track metrics
    rewards.append(total_reward)
    success_rate.append(1 if success else 0)
    steps_per_episode.append(steps)

    # Print progress every 100 episodes
    if episode % 100 == 0 or episode == EPISODES - 1:
        avg_reward = np.mean(rewards[-100:]) if episode >= 100 else np.mean(rewards)
        success_percent = np.mean(success_rate[-100:]) * 100 if episode >= 100 else np.mean(success_rate) * 100
        print(f"Episode {episode:4d} | "
              f"Avg Reward: {avg_reward:6.2f} | "
              f"Success: {success_percent:5.1f}% | "
              f"ε: {epsilon:.3f} | "
              f"Steps: {steps:3d}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

# Plot training metrics
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.ylabel('Total Reward')

plt.subplot(3, 1, 2)
# Smooth success rate with moving average
window_size = 50
success_smooth = np.convolve(success_rate, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(success_rate)), success_smooth)
plt.title('Success Rate (50-episode moving average)')
plt.ylabel('Success Rate')

plt.subplot(3, 1, 3)
# Smooth steps with moving average
steps_smooth = np.convolve(steps_per_episode, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(steps_per_episode)), steps_smooth)
plt.title('Steps per Episode (50-episode moving average)')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()

# Final demonstration with optimal path
state = (0, 0)
path = [state]
running = True
success = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if maze[state] == 3:  # Reached goal
        draw_maze(state, "Final - Success!", path, is_final=True, fps=10)
        success = True
        continue

    if not success:
        action = np.argmax(q_table[state[0], state[1]])
        next_state = list(state)

        if action == 0:  # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Right
            next_state[1] = min(MAZE_SIZE - 1, state[1] + 1)
        elif action == 2:  # Down
            next_state[0] = max(0, min(MAZE_SIZE - 1, state[0] + 1))
        elif action == 3:  # Left
            next_state[1] = max(0, state[1] - 1)

        next_state = tuple(next_state)

        if next_state != state:
            state = next_state
            path.append(state)

        draw_maze(state, "Final Path", path, is_final=True, fps=10)
        time.sleep(0.3)  # Slow down for visualization

pygame.quit()
