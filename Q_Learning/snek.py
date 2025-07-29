# import turtle
# import time
# import random
# import numpy as np
#
# # Set up the screen
# S = turtle.Screen()
# S.setup(width=630, height=630)
# S.bgcolor("Light Green")
# S.title("Snakeeyy")
# S.tracer(0)
#
# # Initialize snake parts
# parts = []
#
# def snake():
#     nPart = turtle.Turtle("square")
#     nPart.penup()
#     nPart.shapesize(0.9)
#     nPart.color("Dark Grey")
#     nPart.goto((-20 * len(parts), 0))
#     parts.append(nPart)
#     nPart.hideturtle()
#
# # Food setup
# noball = 1
# sBall = turtle.Turtle("circle")
# sBall.color("LightCoral")
# sBall.shapesize(1)
# sBall.penup()
# sBall.hideturtle()
#
# # Score display
# writer = turtle.Turtle()
# writer.turtlesize(10)
# writer.hideturtle()
# writer.penup()
#
# score = 0
# try:
#     with open("Highscore.txt", "r") as file:
#         content = file.read().strip()
#         HS = int(content) if content else 0
# except (ValueError, FileNotFoundError):
#     HS = 0
#
# def updateScore(v):
#     global score, HS
#     writer.goto(0, turtle.window_height() / 2 - 50)
#     writer.clear()
#     if v == "over":
#         writer.color("Red")
#         writer.write(f"Game Over! Score: {score}", align="center", font=("Arial", 24, "bold"))
#         return
#     score += v
#     if score > HS:
#         HS = score
#     writer.write(f"Score: {score}  High Score: {HS}", align="center", font=("Arial", 24, "bold"))
#
# updateScore(0)
#
# # Q-Learning Setup
# # State: (dx_bin, dy_bin, danger_up, danger_left, danger_down, danger_right)
# # dx_bin, dy_bin: -2, -1, 0, +1, +2 (finer bins for food location)
# # danger_X: 0=safe, 1=unsafe (body collision in next step)
# actions = [90, 180, 270, 0]  # Up, Left, Down, Right
# q_table = {}  # Dictionary for Q-table
# alpha = 0.2   # Learning rate (increased for faster learning)
# gamma = 0.9   # Discount factor
# epsilon = 0.2 # Exploration rate (increased for early episodes)
#
# def bin_distance(d):
#     # Bin distance into -2, -1, 0, +1, +2 based on thresholds
#     threshold1 = 20  # One grid step
#     threshold2 = 100 # Medium distance
#     if abs(d) < threshold1:
#         return 0
#     elif abs(d) < threshold2:
#         return 1 if d > 0 else -1
#     return 2 if d > 0 else -2
#
# def get_state():
#     # Food position relative to head
#     hx, hy = head.position()
#     fx, fy = sBall.position()
#     # Account for wrap-around
#     dx = fx - hx
#     dy = fy - hy
#     if dx > 315: dx -= 630
#     elif dx < -315: dx += 630
#     if dy > 315: dy -= 630
#     elif dy < -315: dy += 630
#     # Bin the distances
#     dx_bin = bin_distance(dx)
#     dy_bin = bin_distance(dy)
#
#     # Danger check: Is next position (20 pixels) occupied by body?
#     danger = [0, 0, 0, 0]  # Up, Left, Down, Right
#     next_pos = [
#         (hx, hy + 20),  # Up
#         (hx - 20, hy),  # Left
#         (hx, hy - 20),  # Down
#         (hx + 20, hy)   # Right
#     ]
#     for i, pos in enumerate(next_pos):
#         px, py = pos
#         # Apply wrap-around
#         if px > 301: px -= 600
#         elif px < -301: px += 600
#         if py > 301: py -= 600
#         elif py < -301: py += 600
#         for part in parts[1:]:
#             if part.distance(px, py) < 10:
#                 danger[i] = 1
#                 break
#     return (dx_bin, dy_bin, danger[0], danger[1], danger[2], danger[3])
#
# def get_path_action():
#     # Greedy pathfinding: Choose action that reduces Manhattan distance to food
#     hx, hy = head.position()
#     fx, fy = sBall.position()
#     # Account for wrap-around
#     dx = fx - hx
#     dy = fy - hy
#     if dx > 315: dx -= 630
#     elif dx < -315: dx += 630
#     if dy > 315: dy -= 630
#     elif dy < -315: dy += 630
#     # Compute current Manhattan distance
#     current_dist = abs(dx) + abs(dy)
#     # Evaluate each action
#     action_dists = []
#     for i, action in enumerate(actions):
#         if (action + 180) % 360 == head.heading() % 360:
#             action_dists.append(float('inf'))  # Invalid action (reverse)
#             continue
#         if action == 90:  # Up
#             new_pos = (hx, hy + 20)
#             new_dx, new_dy = dx, dy - 20
#         elif action == 180:  # Left
#             new_pos = (hx - 20, hy)
#             new_dx, new_dy = dx + 20, dy
#         elif action == 270:  # Down
#             new_pos = (hx, hy - 20)
#             new_dx, new_dy = dx, dy + 20
#         else:  # Right
#             new_pos = (hx + 20, hy)
#             new_dx, new_dy = dx - 20, dy
#         # Apply wrap-around
#         px, py = new_pos
#         if px > 301: px -= 600
#         elif px < -301: px += 600
#         if py > 301: py -= 600
#         elif py < -301: py += 600
#         # Check if safe
#         safe = True
#         for part in parts[1:]:
#             if part.distance(px, py) < 10:
#                 safe = False
#                 break
#         if not safe:
#             action_dists.append(float('inf'))
#         else:
#             # Compute new Manhattan distance
#             if new_dx > 315: new_dx -= 630
#             elif new_dx < -315: new_dx += 630
#             if new_dy > 315: new_dy -= 630
#             elif new_dy < -315: new_dy += 630
#             new_dist = abs(new_dx) + abs(new_dy)
#             action_dists.append(new_dist)
#     # Choose action with minimum distance, or random safe action if none improve
#     min_dist = min(action_dists)
#     if min_dist == float('inf'):
#         # No safe distance-reducing action; choose any safe action
#         safe_actions = [a for i, a in enumerate(actions) if action_dists[i] != float('inf')]
#         return random.choice(safe_actions) if safe_actions else random.choice(actions)
#     min_indices = [i for i, d in enumerate(action_dists) if d == min_dist]
#     return actions[random.choice(min_indices)]
#
# def choose_action(state):
#     if random.random() < epsilon:
#         return random.choice([a for a in actions if (a + 180) % 360 != head.heading() % 360])
#     if state not in q_table:
#         q_table[state] = [0.0] * 4
#     # Filter valid actions (no reverse)
#     valid_actions = [i for i, a in enumerate(actions) if (a + 180) % 360 != head.heading() % 360]
#     q_values = [q_table[state][i] for i in valid_actions]
#     max_q = max(q_values)
#     max_indices = [i for i, q in zip(valid_actions, q_values) if q == max_q]
#     return actions[random.choice(max_indices)]
#
# def update_q_table(state, action, reward, next_state):
#     if state not in q_table:
#         q_table[state] = [0.0] * 4
#     if next_state not in q_table:
#         q_table[next_state] = [0.0] * 4
#     action_idx = actions.index(action)
#     current_q = q_table[state][action_idx]
#     next_max_q = max(q_table[next_state])
#     q_table[state][action_idx] = current_q + alpha * (reward + gamma * next_max_q - current_q)
#
# # Game loop with RL
# episodes = 1000  # Number of training episodes
# for episode in range(episodes):
#     # Reset game
#     score = 0
#     # Clear previous snake
#     for part in parts:
#         part.clear()
#         part.hideturtle()
#     parts.clear()
#     for _ in range(3):
#         snake()
#     head = parts[0]
#     head.setpos(0, 0)
#     head.setheading(0)
#     noball = 1
#     sBall.hideturtle()
#     updateScore(0)
#     game = True
#     steps = 0
#     max_steps = 1000  # Prevent infinite loops
#
#     while game and steps < max_steps:
#         S.update()
#         time.sleep(0.05)
#         steps += 1
#
#         # Get current state
#         state = get_state()
#
#         # Choose action
#         action = choose_action(state)
#         # Check if action aligns with pathfinding
#         path_action = get_path_action()
#         reward = -0.1  # Default step penalty
#         if action == path_action:
#             reward += 0.5  # Bonus for following path to food
#
#         head.setheading(action)
#
#         # Move snake
#         for p in range(len(parts) - 1, 0, -1):
#             parts[p].showturtle()
#             parts[p].goto(parts[p - 1].position())
#             parts[p].setheading(parts[p - 1].heading())
#         head.showturtle()
#         head.fd(20)
#
#         # Wrap-around boundaries
#         if head.xcor() > 301:
#             head.goto(head.xcor() - 600, head.ycor())
#         elif head.xcor() < -301:
#             head.goto(head.xcor() + 600, head.ycor())
#         elif head.ycor() > 301:
#             head.goto(head.xcor(), head.ycor() - 600)
#         elif head.ycor() < -301:
#             head.goto(head.xcor(), head.ycor() + 600)
#
#         # Check for food
#         if head.distance(sBall) <= 20:
#             noball = 1
#             sBall.hideturtle()
#             updateScore(10)
#             reward = 10
#             snake()
#         else:
#             reward = reward  # Keep path-guided reward or step penalty
#
#         # Spawn food if needed
#         if noball:
#             noball = 0
#             sBall.showturtle()
#             sBall.goto(random.randint(-280, 280), random.randint(-280, 280))
#
#         # Check for self-collision
#         for p in range(3, len(parts)):
#             if head.distance(parts[p]) < 20:
#                 updateScore("over")
#                 reward = -100
#                 game = False
#
#         # Update Q-table
#         next_state = get_state()
#         update_q_table(state, action, reward, next_state)
#
#         if not game:
#             break
#
#     # Save high score
#     with open("Highscore.txt", "w") as file:
#         file.write(str(HS))
#
#     # Reduce epsilon for less exploration over time
#     epsilon = max(0.01, epsilon * 0.995)
#
# S.exitonclick()

import turtle
import time
import random
import numpy as np

# Set up the screen
S = turtle.Screen()
S.setup(width=630, height=630)
S.bgcolor("Light Green")
S.title("Snakeeyy")
S.tracer(0)

# Initialize snake parts
parts = []

def snake():
    nPart = turtle.Turtle("square")
    nPart.penup()
    nPart.shapesize(0.9)
    nPart.color("Dark Grey")
    nPart.goto((-20 * len(parts), 0))
    parts.append(nPart)
    nPart.hideturtle()

# Food setup
noball = 1
sBall = turtle.Turtle("circle")
sBall.color("LightCoral")
sBall.shapesize(1)
sBall.penup()
sBall.hideturtle()

# Score display
writer = turtle.Turtle()
writer.turtlesize(10)
writer.hideturtle()
writer.penup()

score = 0
try:
    with open("Highscore.txt", "r") as file:
        content = file.read().strip()
        HS = int(content) if content else 0
except (ValueError, FileNotFoundError):
    HS = 0

def updateScore(v):
    global score, HS
    writer.goto(0, turtle.window_height() / 2 - 50)
    writer.clear()
    if v == "over":
        writer.color("Red")
        writer.write(f"Game Over! Score: {score}", align="center", font=("Arial", 24, "bold"))
        return
    score += v
    if score > HS:
        HS = score
    writer.write(f"Score: {score}  High Score: {HS}", align="center", font=("Arial", 24, "bold"))

updateScore(0)

# Q-Learning Setup
# State: (dx_bin, dy_bin, danger_up, danger_left, danger_down, danger_right)
# dx_bin, dy_bin: -2, -1, 0, +1, +2 (finer bins for food location)
# danger_X: 0=safe, 1=unsafe (body collision in next step)
actions = [90, 180, 270, 0]  # Up, Left, Down, Right
q_table = {}  # Dictionary for Q-table
alpha = 0.2   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.2 # Exploration rate

def bin_distance(d):
    # Bin distance into -2, -1, 0, +1, +2 based on thresholds
    threshold1 = 20  # One grid step
    threshold2 = 100 # Medium distance
    if abs(d) < threshold1:
        return 0
    elif abs(d) < threshold2:
        return 1 if d > 0 else -1
    return 2 if d > 0 else -2

def get_state():
    # Food position relative to head
    hx, hy = head.position()
    fx, fy = sBall.position()
    # Account for wrap-around
    dx = fx - hx
    dy = fy - hy
    if dx > 315: dx -= 630
    elif dx < -315: dx += 630
    if dy > 315: dy -= 630
    elif dy < -315: dy += 630
    # Bin the distances
    dx_bin = bin_distance(dx)
    dy_bin = bin_distance(dy)

    # Danger check: Is next position (20 pixels) occupied by body?
    danger = [0, 0, 0, 0]  # Up, Left, Down, Right
    next_pos = [
        (hx, hy + 20),  # Up
        (hx - 20, hy),  # Left
        (hx, hy - 20),  # Down
        (hx + 20, hy)   # Right
    ]
    for i, pos in enumerate(next_pos):
        px, py = pos
        # Apply wrap-around
        if px > 301: px -= 600
        elif px < -301: px += 600
        if py > 301: py -= 600
        elif py < -301: py += 600
        for part in parts[1:]:
            if part.distance(px, py) < 10:
                danger[i] = 1
                break
    return (dx_bin, dy_bin, danger[0], danger[1], danger[2], danger[3])

def get_path_action():
    # Greedy pathfinding: Choose action that reduces Manhattan distance to food
    hx, hy = head.position()
    fx, fy = sBall.position()
    # Account for wrap-around
    dx = fx - hx
    dy = fy - hy
    if dx > 315: dx -= 630
    elif dx < -315: dx += 630
    if dy > 315: dy -= 630
    elif dy < -315: dy += 620
    # Compute current Manhattan distance
    current_dist = abs(dx) + abs(dy)
    # Evaluate each action
    action_dists = []
    for i, action in enumerate(actions):
        if (action + 180) % 360 == head.heading() % 360:
            action_dists.append(float('inf'))  # Invalid action (reverse)
            continue
        if action == 90:  # Up
            new_pos = (hx, hy + 20)
            new_dx, new_dy = dx, dy - 20
        elif action == 180:  # Left
            new_pos = (hx - 20, hy)
            new_dx, new_dy = dx + 20, dy
        elif action == 270:  # Down
            new_pos = (hx, hy - 20)
            new_dx, new_dy = dx, dy + 20
        else:  # Right
            new_pos = (hx + 20, hy)
            new_dx, new_dy = dx - 20, dy
        # Apply wrap-around
        px, py = new_pos
        if px > 301: px -= 600
        elif px < -301: px += 600
        if py > 301: py -= 600
        elif py < -301: py += 600
        # Check if safe
        safe = True
        for part in parts[1:]:
            if part.distance(px, py) < 10:
                safe = False
                break
        if not safe:
            action_dists.append(float('inf'))
        else:
            # Compute new Manhattan distance
            if new_dx > 315: new_dx -= 630
            elif new_dx < -315: new_dx += 630
            if new_dy > 315: new_dy -= 630
            elif new_dy < -315: new_dy += 630
            new_dist = abs(new_dx) + abs(new_dy)
            action_dists.append(new_dist)
    # Choose action with minimum distance, or random safe action if none improve
    min_dist = min(action_dists)
    if min_dist == float('inf'):
        # No safe distance-reducing action; choose any safe action
        safe_actions = [a for i, a in enumerate(actions) if action_dists[i] != float('inf')]
        return random.choice(safe_actions) if safe_actions else random.choice(actions)
    min_indices = [i for i, d in enumerate(action_dists) if d == min_dist]
    return actions[random.choice(min_indices)]

def choose_action(state):
    if random.random() < epsilon:
        return random.choice([a for a in actions if (a + 180) % 360 != head.heading() % 360])
    if state not in q_table:
        q_table[state] = [0.0] * 4
    # Filter valid actions (no reverse)
    valid_actions = [i for i, a in enumerate(actions) if (a + 180) % 360 != head.heading() % 360]
    q_values = [q_table[state][i] for i in valid_actions]
    max_q = max(q_values)
    max_indices = [i for i, q in zip(valid_actions, q_values) if q == max_q]
    return actions[random.choice(max_indices)]

def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = [0.0] * 4
    if next_state not in q_table:
        q_table[next_state] = [0.0] * 4
    action_idx = actions.index(action)
    current_q = q_table[state][action_idx]
    next_max_q = max(q_table[next_state])
    q_table[state][action_idx] = current_q + alpha * (reward + gamma * next_max_q - current_q)

# Game loop with RL
episodes = 1000  # Number of training episodes
for episode in range(episodes):
    # Reset game
    score = 0
    episode_reward = 0  # Track total reward for the episode
    # Clear previous snake
    for part in parts:
        part.clear()
        part.hideturtle()
    parts.clear()
    for _ in range(3):
        snake()
    head = parts[0]
    head.setpos(0, 0)
    head.setheading(0)
    noball = 1
    sBall.hideturtle()
    updateScore(0)
    game = True
    steps = 0
    max_steps = 1000  # Prevent infinite loops

    while game and steps < max_steps:
        S.update()
        # time.sleep(0.05)
        steps += 1

        # Get current state
        state = get_state()

        # Choose action
        action = choose_action(state)
        # Check if action aligns with pathfinding
        path_action = get_path_action()
        reward = -0.1  # Default step penalty
        if action == path_action:
            reward += 0.5  # Bonus for following path to food

        head.setheading(action)

        # Move snake
        for p in range(len(parts) - 1, 0, -1):
            parts[p].showturtle()
            parts[p].goto(parts[p - 1].position())
            parts[p].setheading(parts[p - 1].heading())
        head.showturtle()
        head.fd(20)

        # Wrap-around boundaries
        if head.xcor() > 301:
            head.goto(head.xcor() - 600, head.ycor())
        elif head.xcor() < -301:
            head.goto(head.xcor() + 600, head.ycor())
        elif head.ycor() > 301:
            head.goto(head.xcor(), head.ycor() - 600)
        elif head.ycor() < -301:
            head.goto(head.xcor(), head.ycor() + 600)

        # Check for food
        if head.distance(sBall) <= 20:
            noball = 1
            sBall.hideturtle()
            updateScore(10)
            reward = 10
            snake()
            # print(f"Episode {episode + 1}: Food eaten at step {steps}, Score: {score}")
        episode_reward += reward

        # Spawn food if needed
        if noball:
            noball = 0
            sBall.showturtle()
            sBall.goto(random.randint(-280, 280), random.randint(-280, 280))

        # Check for self-collision
        for p in range(3, len(parts)):
            if head.distance(parts[p]) < 20:
                updateScore("over")
                reward = -100
                episode_reward += reward
                # print(f"Episode {episode + 1}: Snake collided with itself at step {steps}, Score: {score}")
                game = False

        # Update Q-table
        next_state = get_state()
        update_q_table(state, action, reward, next_state)

        if not game:
            break

    # Print total reward for the episode
    print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Final Score: {score}")

    # Save high score
    with open("Highscore.txt", "w") as file:
        file.write(str(HS))

    # Reduce epsilon for less exploration over time
    epsilon = max(0.01, epsilon * 0.995)

S.exitonclick()