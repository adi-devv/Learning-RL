# import turtle
# import time
# import random
# import math
# from collections import deque
#
# # Set up the screen
# S = turtle.Screen()
# S.setup(width=700, height=750)
# S.bgcolor("Light Green")
# S.title("Snakeeyy")
# S.tracer(0)
#
# # Speed control variables
# current_speed = 0.05
# speed_step = 0.01
# min_speed = 0.01
# max_speed = 0.2
# paused = False
#
# # Speed control UI
# speed_writer = turtle.Turtle()
# speed_writer.hideturtle()
# speed_writer.penup()
# speed_writer.goto(0, -320)
#
#
# def update_speed_display():
#     speed_writer.clear()
#     status = "PAUSED" if paused else f"Speed: {current_speed:.2f}s/step"
#     speed_writer.write(f"{status} (Up/Down: speed, Space: pause)",
#                        align="center", font=("Arial", 12, "normal"))
#
#
# # Keyboard bindings
# def increase_speed():
#     global current_speed
#     current_speed = max(min_speed, current_speed - speed_step)
#     update_speed_display()
#
#
# def decrease_speed():
#     global current_speed
#     current_speed = min(max_speed, current_speed + speed_step)
#     update_speed_display()
#
#
# def toggle_pause():
#     global paused
#     paused = not paused
#     update_speed_display()
#
#
# S.listen()
# S.onkey(increase_speed, "Up")
# S.onkey(decrease_speed, "Down")
# S.onkey(toggle_pause, "space")
#
# # Initialize snake parts
# parts = []
# GRID_SIZE = 20
#
#
# def snake():
#     nPart = turtle.Turtle("square")
#     nPart.penup()
#     nPart.shapesize(0.9)
#     nPart.color("Dark Grey")
#     nPart.goto((-GRID_SIZE * len(parts), 0))
#     parts.append(nPart)
#     nPart.hideturtle()
#
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
# # Reward system
# FOOD_REWARD = 20
# WALL_CROSS_REWARD = 0.3
# STEP_PENALTY = -0.1
# TURN_PENALTY = -0.1
# COLLISION_PENALTY = -1000
# PATH_BONUS = 1.0
# HIGH_SCORE_REWARD = 50
#
#
# def updateScore(v):
#     global score, HS
#     reward = 0
#     writer.goto(0, 300)
#     writer.clear()
#     if v == "over":
#         writer.color("Red")
#         writer.write(f"Game Over! Score: {score}", align="center", font=("Arial", 24, "bold"))
#         return reward
#     score += v
#     if score > HS:
#         HS = score
#         reward += HIGH_SCORE_REWARD
#     writer.write(f"Score: {score}  High Score: {HS}", align="center", font=("Arial", 24, "bold"))
#     return reward
#
#
# updateScore(0)
#
# # Enhanced Pathfinding and Q-Learning
# actions = [90, 180, 270, 0]  # Up, Left, Down, Right
# q_table = {}
# alpha = 0.2
# gamma = 0.9
# epsilon = 0.3
# min_epsilon = 0.01
# epsilon_decay = 0.995
#
#
# def normalize_position(x, y):
#     """Normalize position to grid coordinates"""
#     return round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE
#
#
# def wrap_position(x, y):
#     """Handle wrap-around boundaries"""
#     if x > 301:
#         x -= 600
#     elif x < -301:
#         x += 600
#     if y > 301:
#         y -= 600
#     elif y < -301:
#         y += 600
#     return x, y
#
#
# def get_snake_body_positions():
#     """Get all snake body positions as a set of normalized coordinates"""
#     body_positions = set()
#     for i, part in enumerate(parts):
#         if i == 0:  # Skip head
#             continue
#         pos = normalize_position(part.xcor(), part.ycor())
#         body_positions.add(pos)
#     return body_positions
#
#
# def is_position_safe(x, y, body_positions):
#     """Check if a position is safe (not colliding with snake body)"""
#     norm_pos = normalize_position(x, y)
#     wrapped_pos = wrap_position(*norm_pos)
#
#     # Check direct collision
#     if wrapped_pos in body_positions:
#         return False
#
#     # Check nearby positions due to turtle size
#     for dx in [-GRID_SIZE, 0, GRID_SIZE]:
#         for dy in [-GRID_SIZE, 0, GRID_SIZE]:
#             check_pos = wrap_position(wrapped_pos[0] + dx, wrapped_pos[1] + dy)
#             if check_pos in body_positions:
#                 distance = math.sqrt((wrapped_pos[0] - check_pos[0]) ** 2 + (wrapped_pos[1] - check_pos[1]) ** 2)
#                 if distance < GRID_SIZE * 0.9:  # Allow small tolerance
#                     return False
#
#     return True
#
#
# def find_safest_path(start_pos, food_pos, body_positions):
#     """Enhanced pathfinding with better body collision detection"""
#     visited = set()
#     queue = deque()
#     start_norm = normalize_position(*start_pos)
#     food_norm = normalize_position(*food_pos)
#
#     queue.append((start_norm, []))
#
#     max_iterations = 1000  # Prevent infinite loops
#     iterations = 0
#
#     while queue and iterations < max_iterations:
#         iterations += 1
#         (x, y), path = queue.popleft()
#
#         # Check if we reached the food
#         wrapped_pos = wrap_position(x, y)
#         wrapped_food = wrap_position(*food_norm)
#
#         if abs(wrapped_pos[0] - wrapped_food[0]) < GRID_SIZE / 2 and abs(
#                 wrapped_pos[1] - wrapped_food[1]) < GRID_SIZE / 2:
#             return path
#
#         pos_key = wrapped_pos
#         if pos_key in visited:
#             continue
#         visited.add(pos_key)
#
#         # Explore all four directions
#         directions = [
#             (0, GRID_SIZE, 90),  # Up
#             (0, -GRID_SIZE, 270),  # Down
#             (-GRID_SIZE, 0, 180),  # Left
#             (GRID_SIZE, 0, 0)  # Right
#         ]
#
#         for dx, dy, action in directions:
#             new_x = x + dx
#             new_y = y + dy
#
#             # Check if this position is safe
#             if is_position_safe(new_x, new_y, body_positions):
#                 queue.append(((new_x, new_y), path + [action]))
#
#     return None
#
#
# def get_safe_actions(head_pos, body_positions, current_heading):
#     """Get all safe actions (no immediate collision)"""
#     safe_actions = []
#
#     for action in actions:
#         # Don't reverse direction
#         if (action + 180) % 360 == current_heading % 360:
#             continue
#
#         # Calculate next position
#         if action == 90:  # Up
#             new_pos = (head_pos[0], head_pos[1] + GRID_SIZE)
#         elif action == 180:  # Left
#             new_pos = (head_pos[0] - GRID_SIZE, head_pos[1])
#         elif action == 270:  # Down
#             new_pos = (head_pos[0], head_pos[1] - GRID_SIZE)
#         else:  # Right (0)
#             new_pos = (head_pos[0] + GRID_SIZE, head_pos[1])
#
#         if is_position_safe(new_pos[0], new_pos[1], body_positions):
#             safe_actions.append(action)
#
#     return safe_actions
#
#
# def get_path_action():
#     """Get the best action using pathfinding and safety checks"""
#     head_pos = (head.xcor(), head.ycor())
#     food_pos = (sBall.xcor(), sBall.ycor())
#     body_positions = get_snake_body_positions()
#     current_heading = head.heading()
#
#     # First, try pathfinding
#     path = find_safest_path(head_pos, food_pos, body_positions)
#     if path and len(path) > 0:
#         # Verify the first action is actually safe
#         first_action = path[0]
#         safe_actions = get_safe_actions(head_pos, body_positions, current_heading)
#         if first_action in safe_actions:
#             return first_action
#
#     # If pathfinding fails, choose the safest available action
#     safe_actions = get_safe_actions(head_pos, body_positions, current_heading)
#
#     if not safe_actions:
#         # Emergency: no safe actions, continue current direction
#         return current_heading
#
#     # Choose action that moves toward food
#     food_pos_wrapped = wrap_position(*food_pos)
#     head_pos_wrapped = wrap_position(*head_pos)
#
#     dx = food_pos_wrapped[0] - head_pos_wrapped[0]
#     dy = food_pos_wrapped[1] - head_pos_wrapped[1]
#
#     # Handle wrap-around for direction calculation
#     if abs(dx) > 300:
#         dx = -dx
#     if abs(dy) > 300:
#         dy = -dy
#
#     # Score each safe action based on how well it moves toward food
#     best_action = safe_actions[0]
#     best_score = float('-inf')
#
#     for action in safe_actions:
#         score = 0
#         if action == 90 and dy > 0:  # Up toward food
#             score = dy
#         elif action == 270 and dy < 0:  # Down toward food
#             score = -dy
#         elif action == 180 and dx < 0:  # Left toward food
#             score = -dx
#         elif action == 0 and dx > 0:  # Right toward food
#             score = dx
#
#         if score > best_score:
#             best_score = score
#             best_action = action
#
#     return best_action
#
#
# # Game loop with enhanced collision detection
# episodes = 1000
# for episode in range(episodes):
#     score = 0
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
#     update_speed_display()
#     game = True
#     steps = 0
#     max_steps = 2000
#     last_pos = head.position()
#     last_heading = head.heading()
#     wall_crossings = 0
#     turn_count = 0
#     consecutive_turns = 0
#
#     while game and steps < max_steps:
#         if not paused:
#             S.update()
#             time.sleep(current_speed)
#             steps += 1
#
#             # Track turns
#             current_heading = head.heading()
#             if current_heading != last_heading:
#                 turn_count += 1
#                 consecutive_turns += 1
#             else:
#                 consecutive_turns = 0
#             last_heading = current_heading
#
#             # Get action and calculate reward
#             action = get_path_action()
#             reward = STEP_PENALTY
#
#             # Apply turn penalty for excessive consecutive turns
#             if consecutive_turns > 2:
#                 reward += TURN_PENALTY * (consecutive_turns - 2)
#
#             head.setheading(action)
#
#             # Move snake
#             for p in range(len(parts) - 1, 0, -1):
#                 parts[p].showturtle()
#                 parts[p].goto(parts[p - 1].position())
#                 parts[p].setheading(parts[p - 1].heading())
#             head.showturtle()
#             head.fd(GRID_SIZE)
#
#             # Check for wall crossing
#             current_pos = head.position()
#             if (abs(current_pos[0] - last_pos[0]) > 300 or
#                     abs(current_pos[1] - last_pos[1]) > 300):
#                 wall_crossings += 1
#                 reward += WALL_CROSS_REWARD
#             last_pos = current_pos
#
#             # Wrap-around boundaries
#             if head.xcor() > 301:
#                 head.goto(head.xcor() - 600, head.ycor())
#             elif head.xcor() < -301:
#                 head.goto(head.xcor() + 600, head.ycor())
#             elif head.ycor() > 301:
#                 head.goto(head.xcor(), head.ycor() - 600)
#             elif head.ycor() < -301:
#                 head.goto(head.xcor(), head.ycor() + 600)
#
#             # Check for food
#             if head.distance(sBall) <= GRID_SIZE:
#                 noball = 1
#                 sBall.hideturtle()
#                 reward += updateScore(10) + FOOD_REWARD
#                 snake()
#
#             # Spawn food if needed
#             if noball:
#                 noball = 0
#                 sBall.showturtle()
#                 valid_pos = False
#                 attempts = 0
#                 while not valid_pos and attempts < 100:
#                     attempts += 1
#                     x = random.randint(-14, 14) * GRID_SIZE
#                     y = random.randint(-14, 14) * GRID_SIZE
#                     valid_pos = True
#                     for part in parts:
#                         if part.distance(x, y) < GRID_SIZE * 1.5:
#                             valid_pos = False
#                             break
#                 sBall.goto(x, y)
#
#             # Enhanced self-collision detection
#             head_pos = normalize_position(head.xcor(), head.ycor())
#             head_wrapped = wrap_position(*head_pos)
#
#             for p in range(1, len(parts)):  # Skip head (index 0)
#                 part_pos = normalize_position(parts[p].xcor(), parts[p].ycor())
#                 part_wrapped = wrap_position(*part_pos)
#
#                 distance = math.sqrt((head_wrapped[0] - part_wrapped[0]) ** 2 +
#                                      (head_wrapped[1] - part_wrapped[1]) ** 2)
#
#                 if distance < GRID_SIZE * 0.9:  # Collision threshold
#                     updateScore("over")
#                     reward += COLLISION_PENALTY
#                     game = False
#                     break
#
#             if not game:
#                 break
#         else:
#             S.update()
#             time.sleep(0.1)
#
#     print(f"Episode {episode + 1}: Score: {score}, Wall Crossings: {wall_crossings}, Turns: {turn_count}")
#
#     with open("Highscore.txt", "w") as file:
#         file.write(str(HS))
#
#     epsilon = max(min_epsilon, epsilon * epsilon_decay)
#
# S.exitonclick()

import turtle
import time
import random
import math
from collections import deque
import pickle
import os

# Set up the screen
S = turtle.Screen()
S.setup(width=700, height=750)
S.bgcolor("Light Green")
S.title("Snakeeyy")
S.tracer(0)

# Speed control variables
current_speed = 0.05
speed_step = 0.01
min_speed = 0.01
max_speed = 0.2
paused = False

# Speed control UI
speed_writer = turtle.Turtle()
speed_writer.hideturtle()
speed_writer.penup()
speed_writer.goto(0, -320)


def update_speed_display():
    speed_writer.clear()
    status = "PAUSED" if paused else f"Speed: {current_speed:.2f}s/step"
    speed_writer.write(f"{status} (Up/Down: speed, Space: pause)",
                       align="center", font=("Arial", 12, "normal"))


# Keyboard bindings
def increase_speed():
    global current_speed
    current_speed = max(min_speed, current_speed - speed_step)
    update_speed_display()


def decrease_speed():
    global current_speed
    current_speed = min(max_speed, current_speed + speed_step)
    update_speed_display()


def toggle_pause():
    global paused
    paused = not paused
    update_speed_display()


S.listen()
S.onkey(increase_speed, "Up")
S.onkey(decrease_speed, "Down")
S.onkey(toggle_pause, "space")

# Initialize snake parts
parts = []
GRID_SIZE = 20


def snake():
    nPart = turtle.Turtle("square")
    nPart.penup()
    nPart.shapesize(0.9)
    nPart.color("Dark Grey")
    nPart.goto((-GRID_SIZE * len(parts), 0))
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
writer.hideturtle()
writer.penup()

score = 0
try:
    with open("Highscore.txt", "r") as file:
        content = file.read().strip()
        HS = int(content) if content else 0
except (ValueError, FileNotFoundError):
    HS = 0

# Reward system
FOOD_REWARD = 20
WALL_CROSS_REWARD = 0.3
STEP_PENALTY = -0.1
TURN_PENALTY = -0.1
COLLISION_PENALTY = -1000
DISTANCE_REWARD = 0.5  # Reward for moving closer to food

# Q-Learning parameters
actions = [90, 180, 270, 0]  # Up, Left, Down, Right
q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration rate
min_epsilon = 0.01
epsilon_decay = 0.995


# Load Q-table if exists
def load_q_table():
    global q_table
    if os.path.exists("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)


# Save Q-table
def save_q_table():
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)


# Load Q-table at start
load_q_table()


def normalize_position(x, y):
    """Normalize position to grid coordinates"""
    return round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE


def wrap_position(x, y):
    """Handle wrap-around boundaries"""
    if x > 301:
        x -= 600
    elif x < -301:
        x += 600
    if y > 301:
        y -= 600
    elif y < -301:
        y += 600
    return x, y


def get_snake_body_positions():
    """Get all snake body positions as a set of normalized coordinates"""
    body_positions = set()
    for i, part in enumerate(parts):
        if i == 0:  # Skip head
            continue
        pos = normalize_position(part.xcor(), part.ycor())
        body_positions.add(pos)
    return body_positions


def is_position_safe(x, y, body_positions):
    """Check if a position is safe (not colliding with snake body)"""
    norm_pos = normalize_position(x, y)
    wrapped_pos = wrap_position(*norm_pos)
    if wrapped_pos in body_positions:
        return False
    for dx in [-GRID_SIZE, 0, GRID_SIZE]:
        for dy in [-GRID_SIZE, 0, GRID_SIZE]:
            check_pos = wrap_position(wrapped_pos[0] + dx, wrapped_pos[1] + dy)
            if check_pos in body_positions:
                distance = math.sqrt((wrapped_pos[0] - check_pos[0]) ** 2 + (wrapped_pos[1] - check_pos[1]) ** 2)
                if distance < GRID_SIZE * 0.9:
                    return False
    return True


def find_safest_path(start_pos, food_pos, body_positions):
    """Enhanced pathfinding with better body collision detection"""
    visited = set()
    queue = deque()
    start_norm = normalize_position(*start_pos)
    food_norm = normalize_position(*food_pos)
    queue.append((start_norm, []))
    max_iterations = 5000
    iterations = 0
    while queue and iterations < max_iterations:
        iterations += 1
        (x, y), path = queue.popleft()
        wrapped_pos = wrap_position(x, y)
        wrapped_food = wrap_position(*food_norm)
        if abs(wrapped_pos[0] - wrapped_food[0]) < GRID_SIZE / 2 and abs(
                wrapped_pos[1] - wrapped_food[1]) < GRID_SIZE / 2:
            return path
        pos_key = wrapped_pos
        if pos_key in visited:
            continue
        visited.add(pos_key)
        directions = [
            (0, GRID_SIZE, 90),
            (0, -GRID_SIZE, 270),
            (-GRID_SIZE, 0, 180),
            (GRID_SIZE, 0, 0)
        ]
        for dx, dy, action in directions:
            new_x = x + dx
            new_y = y + dy
            if is_position_safe(new_x, new_y, body_positions):
                queue.append(((new_x, new_y), path + [action]))
    return None


def get_safe_actions(head_pos, body_positions, current_heading):
    """Get all safe actions (no immediate collision)"""
    safe_actions = []
    for action in actions:
        if (action + 180) % 360 == current_heading % 360:
            continue
        if action == 90:
            new_pos = (head_pos[0], head_pos[1] + GRID_SIZE)
        elif action == 180:
            new_pos = (head_pos[0] - GRID_SIZE, head_pos[1])
        elif action == 270:
            new_pos = (head_pos[0], head_pos[1] - GRID_SIZE)
        else:
            new_pos = (head_pos[0] + GRID_SIZE, head_pos[1])
        if is_position_safe(new_pos[0], new_pos[1], body_positions):
            safe_actions.append(action)
    return safe_actions


def get_state(head, food, body_positions):
    """Get state for Q-learning"""
    head_pos = normalize_position(head.xcor(), head.ycor())
    food_pos = normalize_position(food.xcor(), food.ycor())
    head_wrapped = wrap_position(*head_pos)
    food_wrapped = wrap_position(*food_pos)

    dx = food_wrapped[0] - head_wrapped[0]
    dy = food_wrapped[1] - head_wrapped[1]
    if abs(dx) > 300: dx = -dx
    if abs(dy) > 300: dy = -dy

    # Discretize to reduce state space
    dx = 1 if dx > 0 else -1 if dx < 0 else 0
    dy = 1 if dy > 0 else -1 if dy < 0 else 0

    # Check obstacles in adjacent cells
    obstacles = (
        not is_position_safe(head_pos[0], head_pos[1] + GRID_SIZE, body_positions),
        not is_position_safe(head_pos[0], head_pos[1] - GRID_SIZE, body_positions),
        not is_position_safe(head_pos[0] - GRID_SIZE, head_pos[1], body_positions),
        not is_position_safe(head_pos[0] + GRID_SIZE, head_pos[1], body_positions)
    )

    return (dx, dy, obstacles, head.heading())


def get_q_learning_action(head, food, body_positions, current_heading):
    """Choose action using Q-learning with epsilon-greedy policy"""
    global epsilon
    state = get_state(head, food, body_positions)
    safe_actions = get_safe_actions((head.xcor(), head.ycor()), body_positions, current_heading)

    if not safe_actions:
        return current_heading

    # Fallback to pathfinding in early training
    if state not in q_table or all(q_table[state].get(a, 0) == 0 for a in safe_actions):
        path = find_safest_path((head.xcor(), head.ycor()), (food.xcor(), food.ycor()), body_positions)
        if path and len(path) > 0 and path[0] in safe_actions:
            return path[0]
        return random.choice(safe_actions)

    # Epsilon-greedy policy
    if random.random() < epsilon:
        return random.choice(safe_actions)
    else:
        if state not in q_table:
            q_table[state] = {action: 0 for action in actions}
        best_action = safe_actions[0]
        best_q_value = q_table[state].get(best_action, 0)
        for action in safe_actions:
            q_value = q_table[state].get(action, 0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action


def get_distance_to_food(head, food):
    """Calculate Manhattan distance to food with wrap-around"""
    head_pos = normalize_position(head.xcor(), head.ycor())
    food_pos = normalize_position(food.xcor(), food.ycor())
    head_wrapped = wrap_position(*head_pos)
    food_wrapped = wrap_position(*food_pos)
    dx = abs(head_wrapped[0] - food_wrapped[0])
    dy = abs(head_wrapped[1] - food_wrapped[1])
    if dx > 300: dx = 600 - dx
    if dy > 300: dy = 600 - dy
    return dx + dy


def updateScore(v):
    global score, HS
    reward = 0
    writer.goto(0, 300)
    writer.clear()
    if v == "over":
        writer.color("Red")
        writer.write(f"Game Over! Score: {score}", align="center", font=("Arial", 24, "bold"))
        return reward
    score += v
    if score > HS:
        HS = score
        reward += 50  # HIGH_SCORE_REWARD
    writer.write(f"Score: {score}  High Score: {HS}", align="center", font=("Arial", 24, "bold"))
    return reward


# Game loop
episodes = 1000
for episode in range(episodes):
    score = 0
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
    update_speed_display()
    game = True
    steps = 0
    max_steps = 2000
    last_pos = head.position()
    last_heading = head.heading()
    wall_crossings = 0
    turn_count = 0
    consecutive_turns = 0
    last_distance = get_distance_to_food(head, sBall)

    while game and steps < max_steps:
        if not paused:
            if (episode + 1) % 1000 == 0:
                S.update()
                time.sleep(current_speed)
            steps += 1

            # Track turns
            current_heading = head.heading()
            if current_heading != last_heading:
                turn_count += 1
                consecutive_turns += 1
            else:
                consecutive_turns = 0
            last_heading = current_heading

            # Get state and action
            body_positions = get_snake_body_positions()
            current_state = get_state(head, sBall, body_positions)
            action = get_q_learning_action(head, sBall, body_positions, current_heading)

            # Calculate reward
            reward = STEP_PENALTY
            if consecutive_turns > 2:
                reward += TURN_PENALTY * (consecutive_turns - 2)

            # Move snake
            head.setheading(action)
            for p in range(len(parts) - 1, 0, -1):
                parts[p].showturtle()
                parts[p].goto(parts[p - 1].position())
                parts[p].setheading(parts[p - 1].heading())
            head.showturtle()
            head.fd(GRID_SIZE)

            # Check for wall crossing
            current_pos = head.position()
            if abs(current_pos[0] - last_pos[0]) > 300 or abs(current_pos[1] - last_pos[1]) > 300:
                wall_crossings += 1
                reward += WALL_CROSS_REWARD
            last_pos = current_pos

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
            if head.distance(sBall) <= GRID_SIZE:
                noball = 1
                sBall.hideturtle()
                reward += updateScore(10) + FOOD_REWARD
                snake()
                last_distance = get_distance_to_food(head, sBall)

            # Spawn food if needed
            if noball:
                noball = 0
                sBall.showturtle()
                valid_pos = False
                attempts = 0
                while not valid_pos and attempts < 100:
                    attempts += 1
                    x = random.randint(-14, 14) * GRID_SIZE
                    y = random.randint(-14, 14) * GRID_SIZE
                    valid_pos = True
                    for part in parts:
                        if part.distance(x, y) < GRID_SIZE * 1.5:
                            valid_pos = False
                            break
                sBall.goto(x, y)

            # Add distance-based reward
            current_distance = get_distance_to_food(head, sBall)
            if current_distance < last_distance:
                reward += DISTANCE_REWARD
            elif current_distance > last_distance:
                reward -= DISTANCE_REWARD
            last_distance = current_distance

            # Check for collision
            head_pos = normalize_position(head.xcor(), head.ycor())
            head_wrapped = wrap_position(*head_pos)
            for p in range(1, len(parts)):
                part_pos = normalize_position(parts[p].xcor(), parts[p].ycor())
                part_wrapped = wrap_position(*part_pos)
                distance = math.sqrt((head_wrapped[0] - part_wrapped[0]) ** 2 +
                                     (head_wrapped[1] - part_wrapped[1]) ** 2)
                if distance < GRID_SIZE * 0.9:
                    updateScore("over")
                    reward += COLLISION_PENALTY
                    game = False

            # Update Q-table
            next_state = get_state(head, sBall, body_positions)
            if current_state not in q_table:
                q_table[current_state] = {a: 0 for a in actions}
            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in actions}
            current_q = q_table[current_state].get(action, 0)
            max_next_q = max(q_table[next_state].values()) if q_table[next_state] else 0
            q_table[current_state][action] = current_q + alpha * (reward + gamma * max_next_q - current_q)

            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        else:
            S.update()
            time.sleep(0.1)

    print(f"Episode {episode + 1}: Score: {score}, Wall Crossings: {wall_crossings}, Turns: {turn_count}")
    with open("Highscore.txt", "w") as file:
        file.write(str(HS))
    save_q_table()

S.exitonclick()