import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import random
from collections import deque

# Initialize Pygame
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("RL Maze Solver with Moving Walls")
cell_size = width // 10
clock = pygame.time.Clock()

# Constants
MAZE_SIZE = 10
WALL_DENSITY = 0.25
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 300
WALL_MOVE_PROBABILITY = 0.05  # Chance a wall will move each step

# Colors
COLORS = {
    'background': (240, 240, 240),
    'static_wall': (50, 50, 50),
    'moving_wall': (100, 100, 100),  # Different color for moving walls
    'start': (30, 144, 255),
    'goal': (50, 205, 50),
    'agent': (220, 20, 60),
    'path': (255, 165, 0),
    'visited': (173, 216, 230),
    'text': (0, 0, 0),
    'grid': (200, 200, 200)
}


class MovingMaze:
    def __init__(self):
        self.static_maze = np.zeros((MAZE_SIZE, MAZE_SIZE), dtype=int)
        self.moving_walls = np.zeros((MAZE_SIZE, MAZE_SIZE), dtype=int)
        self.initialize_maze()

    def initialize_maze(self):
        # Set start and goal
        self.static_maze[0, 0] = 2  # Start
        self.static_maze[-1, -1] = 3  # Goal

        # Place static walls
        wall_count = 0
        max_walls = int(MAZE_SIZE * MAZE_SIZE * WALL_DENSITY * 0.7)  # 70% static walls

        while wall_count < max_walls:
            i, j = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
            if self.static_maze[i, j] == 0 and (i, j) not in [(0, 0), (MAZE_SIZE - 1, MAZE_SIZE - 1)]:
                self.static_maze[i, j] = 1
                wall_count += 1

        # Place initial moving walls (30% of total walls)
        moving_wall_count = 0
        max_moving_walls = int(MAZE_SIZE * MAZE_SIZE * WALL_DENSITY * 0.3)

        while moving_wall_count < max_moving_walls:
            i, j = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
            if (self.static_maze[i, j] == 0 and self.moving_walls[i, j] == 0 and
                    (i, j) not in [(0, 0), (MAZE_SIZE - 1, MAZE_SIZE - 1)]):
                self.moving_walls[i, j] = 1
                moving_wall_count += 1

        # Ensure path exists by removing walls if needed
        while not self.is_solvable():
            # Remove a random wall (static or moving)
            i, j = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
            if self.static_maze[i, j] == 1:
                self.static_maze[i, j] = 0
            elif self.moving_walls[i, j] == 1:
                self.moving_walls[i, j] = 0

    def is_solvable(self):
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
                if (0 <= ni < MAZE_SIZE and 0 <= nj < MAZE_SIZE and
                        self.static_maze[ni, nj] != 1 and self.moving_walls[ni, nj] != 1):
                    queue.append((ni, nj))
        return False

    def move_walls(self):
        # Create a copy of current moving walls
        new_moving_walls = np.zeros((MAZE_SIZE, MAZE_SIZE), dtype=int)

        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                if self.moving_walls[i, j] == 1:
                    # Decide whether to move this wall
                    if random.random() < WALL_MOVE_PROBABILITY:
                        # Try to move to adjacent cell
                        possible_moves = []
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < MAZE_SIZE and 0 <= nj < MAZE_SIZE and
                                    self.static_maze[ni, nj] == 0 and
                                    self.moving_walls[ni, nj] == 0 and
                                    (ni, nj) not in [(0, 0), (MAZE_SIZE - 1, MAZE_SIZE - 1)]):
                                possible_moves.append((ni, nj))

                        if possible_moves:
                            new_i, new_j = random.choice(possible_moves)
                            new_moving_walls[new_i, new_j] = 1
                        else:
                            new_moving_walls[i, j] = 1  # Stay in place if can't move
                    else:
                        new_moving_walls[i, j] = 1

        self.moving_walls = new_moving_walls

    def get_cell_state(self, i, j):
        if self.static_maze[i, j] == 1:
            return 1  # Static wall
        if self.moving_walls[i, j] == 1:
            return 4  # Moving wall (using 4 as new state)
        return self.static_maze[i, j]  # Start, goal, or empty


# Initialize maze and Q-table
maze = MovingMaze()
q_table = np.random.uniform(low=-0.1, high=0.1, size=(MAZE_SIZE, MAZE_SIZE, 5))  # 5 actions now (including waiting)

# Hyperparameters
alpha = 0.2  # The agent incorporates 20% of new information and keeps 80% of its existing knowledge in each update
gamma = 0.95  # Determines how much the agent values future rewards compared to immediate rewards
epsilon = 1.0  # Exploration Rate
epsilon_min = 0.01
epsilon_decay = 0.995  # Exploration Decay Rate

rewards = []
success_rate = []
steps_per_episode = []


def draw_maze(agent_pos, episode, path, visited=None, is_final=False, fps=60):
    screen.fill(COLORS['background'])

    # Draw visited cells
    if visited and not is_final:
        for i, j in visited:
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, COLORS['visited'], rect)

    # Draw maze elements
    for i in range(MAZE_SIZE):
        for j in range(MAZE_SIZE):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            cell_state = maze.get_cell_state(i, j)

            if cell_state == 1:  # Static wall
                pygame.draw.rect(screen, COLORS['static_wall'], rect)
            elif cell_state == 4:  # Moving wall
                pygame.draw.rect(screen, COLORS['moving_wall'], rect)
            elif cell_state == 2:  # Start
                pygame.draw.rect(screen, COLORS['start'], rect)
            elif cell_state == 3:  # Goal
                pygame.draw.rect(screen, COLORS['goal'], rect)

            pygame.draw.rect(screen, COLORS['grid'], rect, 1)

    # Draw path
    if len(path) >= 2:
        for k in range(1, len(path)):
            i_prev, j_prev = path[k - 1]
            i_curr, j_curr = path[k]
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

    # Draw info
    font = pygame.font.SysFont('Arial', 24)
    episode_text = f'Episode: {episode}' if not is_final else 'Final Path'
    text_surface = font.render(episode_text, True, COLORS['text'])
    screen.blit(text_surface, (10, 10))

    if not is_final:
        epsilon_text = f'ε: {epsilon:.3f}'
        epsilon_surface = font.render(epsilon_text, True, COLORS['text'])
        screen.blit(epsilon_surface, (10, 40))

    pygame.display.flip()
    clock.tick(fps)


# Training loop
start_time = time.time()
print(f"Training started at: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

for episode in range(EPISODES):
    state = (0, 0)
    path = [state]
    visited = set([state])
    total_reward = 0
    done = False
    success = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Move walls before agent takes action
        maze.move_walls()

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(5)  # 0-3: movement, 4: wait
        else:
            action = np.argmax(q_table[state[0], state[1]])

        # Calculate next state
        next_state = list(state)
        if action == 0:  # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Right
            next_state[1] = min(MAZE_SIZE - 1, state[1] + 1)
        elif action == 2:  # Down
            next_state[0] = max(0, min(MAZE_SIZE - 1, state[0] + 1))
        elif action == 3:  # Left
            next_state[1] = max(0, state[1] - 1)
        # Action 4: Wait (no position change)

        next_state = tuple(next_state)

        # Get cell state (including moving walls)
        next_cell_state = maze.get_cell_state(next_state[0], next_state[1])

        # Enhanced reward system
        if next_cell_state == 1 or next_cell_state == 4:  # Hit any wall
            reward = -1
            next_state = state  # Stay in current state
        elif next_cell_state == 3:  # Reached goal
            reward = 10
            done = True
            success = True
        else:
            # Reward based on progress toward goal
            current_dist = (MAZE_SIZE - 1 - state[0]) + (MAZE_SIZE - 1 - state[1])
            next_dist = (MAZE_SIZE - 1 - next_state[0]) + (MAZE_SIZE - 1 - next_state[1])
            reward = 0.5 if next_dist < current_dist else -0.2

            # Penalty for revisiting cells
            if next_state in visited and action != 4:  # No penalty for waiting
                reward -= 0.3

        total_reward += reward
        visited.add(next_state)

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - q_table[state[0], state[1], action]
        q_table[state[0], state[1], action] += alpha * td_error

        # Update state and path
        if next_state != state:
            state = next_state
            path.append(state)

        # Visualize
        if episode % 10 == 0 or episode >= EPISODES - 10:
            draw_maze(state, episode, path, visited)

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track metrics
    rewards.append(total_reward)
    success_rate.append(1 if success else 0)
    steps_per_episode.append(steps)

    # Print progress
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
window_size = 50
success_smooth = np.convolve(success_rate, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(success_rate)), success_smooth)
plt.title('Success Rate (50-episode moving average)')
plt.ylabel('Success Rate')

plt.subplot(3, 1, 3)
steps_smooth = np.convolve(steps_per_episode, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(steps_per_episode)), steps_smooth)
plt.title('Steps per Episode (50-episode moving average)')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.savefig('training_metrics_moving_walls.png', dpi=300)
plt.show()

# Final demonstration
state = (0, 0)
path = [state]
running = True
success = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if maze.get_cell_state(state[0], state[1]) == 3:
        draw_maze(state, "Final - Success!", path, is_final=True, fps=10)
        success = True
        continue

    if not success:
        # Move walls in final demo too
        maze.move_walls()

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
        next_cell_state = maze.get_cell_state(next_state[0], next_state[1])

        # Only move if not hitting a wall
        if next_cell_state != 1 and next_cell_state != 4:
            state = next_state
            path.append(state)

        draw_maze(state, "Final Path", path, is_final=True, fps=10)
        time.sleep(0.3)

pygame.quit()
