#ORIGINAL GAME

import turtle, time, random

T = turtle.Turtle
S = turtle.Screen()
S.setup(width=630, height=630)
S.bgcolor("Light Green")
S.title("Snakeeyy")
S.tracer(0)
parts = []


# add moving smol snakea
def snake():
    nPart = T("square")
    nPart.penup()
    nPart.shapesize(0.9)
    nPart.color("Dark Grey")
    nPart.goto((-20 * _, 0))
    parts.append(nPart)
    nPart.hideturtle()


def dir(a):
    if (a + 180) % 360 != head.heading() % 360:
        head.setheading(a)


for _ in range(3):
    snake()
head = parts[0]

noball = 1
sBall = T("circle")
sBall.color("LightCoral")
sBall.shapesize(1)
sBall.penup()
sBall.hideturtle()

writer = T()
writer.turtlesize(10)
writer.hideturtle()
writer.penup()

score = 0
file = open("Highscore.txt","r+")
HS = int(file.read())


def updateScore(v):
    global score, HS
    writer.goto(0, turtle.window_height() / 2 - 50)
    writer.clear()
    if v == "over":
        writer.color("Red")
        writer.write(f"Score: Snake bit itself!", align="center", font="Arial,24,Bold")
        return

    score += v
    if score > HS:
        HS = score
    writer.write(f"Score: {score}  High Score: {HS}", align="center", font="Arial,24,Bold")


updateScore(0)

S.listen()
S.onkey(lambda: dir(90), 'Up')
S.onkey(lambda: dir(180), 'Left')
S.onkey(lambda: dir(270), 'Down')
S.onkey(lambda: dir(0), 'Right')

game = True

while game:
    S.update()
    time.sleep(.05)
    for p in range(0, len(parts)):
        parts[p].showturtle()
        if p != 0:
            parts[len(parts) - p].goto(parts[len(parts) - p - 1].position())
            parts[len(parts) - p].setheading(parts[len(parts) - p - 1].heading())

    for p in range(3, len(parts)):
        if head.distance(parts[p]) < 20:
            updateScore("over")
            game = False

    head.fd(20)

    if noball:
        noball = 0
        sBall.showturtle()
        sBall.goto(random.randint(-280, 280), random.randint(-280, 280))

    if head.distance(sBall) <= 20:
        noball = 1
        sBall.hideturtle()
        updateScore(10)
        snake()

    if head.xcor() > 301:
        head.goto(-300, head.ycor())
    elif head.xcor() < -301:
        head.goto(300, head.ycor())
    elif head.ycor() > 301:
        head.goto(head.xcor(), -300)
    elif head.ycor() < -301:
        head.goto(head.xcor(), 300)

file.seek(0)
file.truncate()
file.write(str(HS))
file.close()
S.exitonclick()
