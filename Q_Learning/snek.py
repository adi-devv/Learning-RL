import turtle
import time
import random
import math
from collections import deque

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
PATH_BONUS = 1.0
HIGH_SCORE_REWARD = 50  # New reward for beating high score

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
        reward += HIGH_SCORE_REWARD  # Reward for beating high score
    writer.write(f"Score: {score}  High Score: {HS}", align="center", font=("Arial", 24, "bold"))
    return reward

updateScore(0)

# Enhanced Pathfinding and Q-Learning
actions = [90, 180, 270, 0]  # Up, Left, Down, Right
q_table = {}
alpha = 0.2
gamma = 0.9
epsilon = 0.3
min_epsilon = 0.01
epsilon_decay = 0.995

def get_snake_body_positions():
    return {(part.xcor(), part.ycor()) for part in parts[1:]}

def find_safest_path(start_pos, food_pos, body_positions):
    grid_size = 20
    visited = set()
    queue = deque()
    queue.append((start_pos, []))
    body_grid = {(round(x / grid_size), round(y / grid_size)) for (x, y) in body_positions}

    while queue:
        (x, y), path = queue.popleft()
        if (round(x), round(y)) == (round(food_pos[0]), round(food_pos[1])):
            return path
        if (round(x), round(y)) in visited:
            continue
        visited.add((round(x), round(y)))

        for dx, dy, action in [(0, grid_size, 90), (0, -grid_size, 270),
                               (-grid_size, 0, 180), (grid_size, 0, 0)]:
            new_x = x + dx
            new_y = y + dy
            if new_x > 301:
                new_x -= 600
            elif new_x < -301:
                new_x += 600
            if new_y > 301:
                new_y -= 600
            elif new_y < -301:
                new_y += 600

            grid_pos = (round(new_x / grid_size), round(new_y / grid_size))
            if grid_pos not in body_grid:
                queue.append(((new_x, new_y), path + [action]))
    return None

def get_path_action():
    head_pos = (head.xcor(), head.ycor())
    food_pos = (sBall.xcor(), sBall.ycor())
    body_positions = get_snake_body_positions()

    path = find_safest_path(head_pos, food_pos, body_positions)
    if path and len(path) > 0:
        return path[0]

    safe_actions = []
    current_heading = head.heading()
    for action in actions:
        if (action + 180) % 360 == current_heading % 360:
            continue
        if action == 90:
            new_pos = (head_pos[0], head_pos[1] + 20)
        elif action == 180:
            new_pos = (head_pos[0] - 20, head_pos[1])
        elif action == 270:
            new_pos = (head_pos[0], head_pos[1] - 20)
        else:
            new_pos = (head_pos[0] + 20, head_pos[1])

        px, py = new_pos
        if px > 301:
            px -= 600
        elif px < -301:
            px += 600
        if py > 301:
            py -= 600
        elif py < -301:
            py += 600

        safe = True
        for (bx, by) in body_positions:
            if math.sqrt((px - bx) ** 2 + (py - by) ** 2) < 15:
                safe = False
                break
        if safe:
            safe_actions.append(action)

    if safe_actions:
        dx = food_pos[0] - head_pos[0]
        dy = food_pos[1] - head_pos[1]
        if dx > 315:
            dx -= 630
        elif dx < -315:
            dx += 630
        if dy > 315:
            dy -= 630
        elif dy < -315:
            dy += 630

        action_scores = []
        for action in safe_actions:
            if action == 90:
                score = dy
            elif action == 180:
                score = -dx
            elif action == 270:
                score = -dy
            else:
                score = dx
            action_scores.append(score)
        return safe_actions[action_scores.index(max(action_scores))]
    return current_heading

# Game loop with enhanced rewards
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

    while game and steps < max_steps:
        if not paused:
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

            # Get action and calculate reward
            state = (round(head.xcor() / 20), round(head.ycor() / 20),
                     round(sBall.xcor() / 20), round(sBall.ycor() / 20),
                     *[1 if (part.xcor(), part.ycor()) in get_snake_body_positions() else 0
                       for part in parts[1:]])

            action = get_path_action()
            reward = STEP_PENALTY

            # Apply turn penalty for excessive consecutive turns
            if consecutive_turns > 2:
                reward += TURN_PENALTY * (consecutive_turns - 2)

            head.setheading(action)

            # Move snake
            for p in range(len(parts) - 1, 0, -1):
                parts[p].showturtle()
                parts[p].goto(parts[p - 1].position())
                parts[p].setheading(parts[p - 1].heading())
            head.showturtle()
            head.fd(20)

            # Check for wall crossing
            current_pos = head.position()
            if (abs(current_pos[0] - last_pos[0]) > 300 or
                    abs(current_pos[1] - last_pos[1]) > 300):
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
            if head.distance(sBall) <= 20:
                noball = 1
                sBall.hideturtle()
                reward += updateScore(10) + FOOD_REWARD
                snake()

            # Spawn food if needed
            if noball:
                noball = 0
                sBall.showturtle()
                valid_pos = False
                while not valid_pos:
                    x = random.randint(-280, 280)
                    y = random.randint(-280, 280)
                    valid_pos = True
                    for part in parts:
                        if part.distance(x, y) < 30:
                            valid_pos = False
                            break
                sBall.goto(x, y)

            # Check for self-collision
            for p in range(3, len(parts)):
                if head.distance(parts[p]) < 20:
                    updateScore("over")
                    reward += COLLISION_PENALTY
                    game = False

            # Update Q-table
            next_state = (round(head.xcor() / 20), round(head.ycor() / 20),
                          round(sBall.xcor() / 20), round(sBall.ycor() / 20),
                          *[1 if (part.xcor(), part.ycor()) in get_snake_body_positions() else 0
                            for part in parts[1:]])

            if state not in q_table:
                q_table[state] = [0.0] * 4
            if next_state not in q_table:
                q_table[next_state] = [0.0] * 4

            action_idx = actions.index(action)
            max_next_q = max(q_table[next_state])
            q_table[state][action_idx] = (1 - alpha) * q_table[state][action_idx] + \
                                         alpha * (reward + gamma * max_next_q)

            if not game:
                break
        else:
            S.update()
            time.sleep(0.1)

    print(f"Episode {episode + 1}: Score: {score}, Wall Crossings: {wall_crossings}, Turns: {turn_count}")

    with open("Highscore.txt", "w") as file:
        file.write(str(HS))

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

S.exitonclick()