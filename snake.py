import copy
import os
from turtle import *
from random import *
from pathfinding import GameGraph, pathfinding


# 常量区
LEFT = -40
RIGHT = 40
TOP = 40
BUTTON = -40

# 核心
game_graph = None

def square(x, y, size, sq_color):
    """绘制小正方形, 代表一格"""
    penup()
    goto(x, y)
    color(sq_color)
    pendown()

    begin_fill()
    for i in range(4):
        forward(size)
        left(90)
    end_fill()


def frame():
    """绘制边框"""
    for i in range(LEFT - 10, RIGHT + 20, 10):
        square(i, BUTTON - 10, 10, 'gray')
        square(i, TOP + 10, 10, 'gray')
    for i in range(BUTTON - 10, TOP + 20, 10):
        square(LEFT - 10, i, 10, 'gray')
        square(RIGHT + 10, i, 10, 'gray')


def change(direction):
    """改变蛇的运动方向"""
    global aim_x, aim_y
    x, y = direction
    if x != -aim_x or y != -aim_y:
        aim_x = x
        aim_y = y


def inside(head_x, head_y):
    """判断蛇是否在边框内"""
    if -210 < head_x < 190 and -200 < head_y < 200:
        return True
    else:
        return False


all_food = []                  # 所有食物的位置
for x in range(LEFT, RIGHT + 10, 10):
    for y in range(BUTTON, TOP + 10, 10):
        all_food.append((x, y))


def new_food():
    """随机生成食物位置"""
    # todo 优化一下食物生成算法
    food = all_food.copy()
    for i in snake:            # 去掉蛇所在的位置
        food.remove(i)
    new_food_x, new_food_y = food.pop(randint(0, len(food) - 1))
    return new_food_x, new_food_y


snake = [(0, 0), (10, 0)]               # 蛇的起始位置
food_x, food_y = new_food()    # 食物的起始位置
aim_x, aim_y = 0, 0            # 蛇的起始方向


def move():
    global food_x, food_y

    change(pathfinding(game_graph))
    # 同步一下 game_graph 的状态
    game_graph.set_aim(aim_x, aim_y)
    game_graph.move_snake()

    head_move_x = snake[-1][0] + aim_x
    head_move_y = snake[-1][1] + aim_y

    # 判断是否撞到边框或者撞到自己
    # 此处也有修改，去除了原程序中对于尾部的错误判断
    if not inside(head_move_x, head_move_y) or ((head_move_x, head_move_y) in snake and (head_move_x,head_move_y) != snake[0]):
        square(head_move_x, head_move_y, 10, 'red')
        update()
        print('得分: ', len(snake))
        return

    snake.append((head_move_x, head_move_y))

    # 判断是否吃到食物以及是否胜利
    if head_move_x == food_x and head_move_y == food_y:
        if len(snake) >= len(all_food)*19//20:     # 限制松一点
            print('YOU WIN!')
            return
        else:
            food_x, food_y = new_food()
            # 更新食物同时要更新蛇蛇，更新在 game_graph.snake 里
            game_graph.set_food((food_x, food_y))
    else:
        snake.pop(0)

    clear()

    # 绘制边框, 蛇和食物
    frame()
    for body in snake:
        square(body[0], body[1], 10, 'black')
    square(snake[-1][0], snake[-1][1], 10, 'red')
    square(food_x, food_y, 10, 'green')
    update()

    ontimer(move)


def start(left=-40, right=40, top=40, button=-40):
    global LEFT, RIGHT, TOP, BUTTON
    global game_graph
    LEFT = left
    RIGHT = right
    TOP = top
    BUTTON = button
    os.system("pause")
    setup(420, 420)
    title('贪吃蛇')
    hideturtle()
    tracer(False)
    # 此处传入的是蛇蛇的深复制
    game_graph = GameGraph(copy.deepcopy(snake), (food_x, food_y), {'xmin': LEFT, 'xmax': RIGHT, 'ymin': BUTTON, 'ymax': TOP})
    move()
    done()


if __name__ == '__main__':
    start()
