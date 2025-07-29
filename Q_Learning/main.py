import numpy as np
import pygame
import time
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("RL Maze Solver")
cell_size = width // 10  # Adjusted for 10x10 maze

# Define maze (0: free, 1: wall, 2: start, 3: goal)
maze = np.array([
    [2, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 1, 1, 0, 3]
])

# Q-learning parameters
q_table = np.zeros((10, 10, 4))  # 10x10 maze, 4 actions (up, right, down, left)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.05  # Reduced exploration rate
episodes = 2000  # Increased for larger maze
rewards = []  # Track rewards for plotting

# Colors
WHITE, BLACK, RED, GREEN, BLUE = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)


def draw_maze(agent_pos, episode, path, is_final=False):
    screen.fill(WHITE)
    # Draw maze
    for i in range(10):
        for j in range(10):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            if maze[i, j] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif maze[i, j] == 2:
                pygame.draw.rect(screen, BLUE, rect)
            elif maze[i, j] == 3:
                pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Grid lines
    # Draw path as a red line if 2+ points exist
    if len(path) >= 2:
        path_points = [(pos[1] * cell_size + cell_size // 2, pos[0] * cell_size + cell_size // 2) for pos in path]
        pygame.draw.lines(screen, RED, False, path_points, 3)  # Thinner line
    # Draw agent
    agent_rect = pygame.Rect(agent_pos[1] * cell_size + 5, agent_pos[0] * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.ellipse(screen, RED, agent_rect)
    # Draw episode label
    font = pygame.font.SysFont(None, 24)
    text = font.render(f'Episode: {episode}' if not is_final else 'Final Path', True, BLACK)
    screen.blit(text, (10, 10))
    pygame.display.flip()


# Training with visualization and timing
start_time = time.time()
print(f"Training started at: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
for episode in range(episodes):
    episode_start = time.time()
    state = (0, 0)  # Start position
    path = [state]  # Track path for this episode
    total_reward = 0
    done = False
    draw_maze(state, episode, path)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Choose action with decaying epsilon
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])
        epsilon *= 0.995  # Decay epsilon

        # Calculate next state with explicit adjacent check
        next_state = state
        if action == 0 and state[0] > 0 and (state[0] - 1, state[1]) != state:  # Up
            next_state = (state[0] - 1, state[1])
        elif action == 1 and state[1] < 9 and (state[0], state[1] + 1) != state:  # Right
            next_state = (state[0], state[1] + 1)
        elif action == 2 and state[0] < 9 and (state[0] + 1, state[1]) != state:  # Down
            next_state = (state[0] + 1, state[1])
        elif action == 3 and state[1] > 0 and (state[0], state[1] - 1) != state:  # Left
            next_state = (state[0], state[1] - 1)

        # Get reward
        if maze[next_state] == 1:  # Wall
            reward = -10
            next_state = state
        elif maze[next_state] == 3:  # Goal
            reward = 100
            done = True
        else:
            reward = -1
        total_reward += reward

        # Update Q-table
        q_table[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action]
        )

        state = next_state
        if state != path[-1]:  # Append only if moved to a new adjacent cell
            path.append(state)
        draw_maze(state, episode, path)
        time.sleep(0.02)  # Slightly slower for visibility
    rewards.append(total_reward)
    episode_duration = time.time() - episode_start
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Duration: {episode_duration:.2f} seconds")
end_time = time.time()
print(f"Training ended at: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
print(f"Total training time: {(end_time - start_time):.2f} seconds")

# Plot rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Agent Learning Progress')
plt.savefig('rewards_plot.png')  # Save for capstone report
plt.show()

# Demo the trained agent
state = (0, 0)
path = [state]
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if maze[state] == 3:  # Stop at goal
        draw_maze(state, 'Final', path, is_final=True)
        continue
    action = np.argmax(q_table[state[0], state[1]])
    draw_maze(state, 'Final', path, is_final=True)
    time.sleep(0.5)  # Slower for final path
    if action == 0 and state[0] > 0 and (state[0] - 1, state[1]) != state:
        state = (state[0] - 1, state[1])
    elif action == 1 and state[1] < 9 and (state[0], state[1] + 1) != state:
        state = (state[0], state[1] + 1)
    elif action == 2 and state[0] < 9 and (state[0] + 1, state[1]) != state:
        state = (state[0] + 1, state[1])
    elif action == 3 and state[1] > 0 and (state[0], state[1] - 1) != state:
        state = (state[0], state[1] - 1)
    if state != path[-1]:
        path.append(state)
pygame.quit()