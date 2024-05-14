import copy
import turtle
import random
from GameGraph import GameGraph
from pathfinding_heuristics import pathfinding

# 常量区
LEFT, RIGHT, TOP, BOTTOM = 0, 0, 0, 0
all_food = []                  # 所有食物的位置
SIZE = None   # 每个方块的尺寸
LOOSE = None   # 达到 LOOSE*全部方格数 就算赢
MAX_REFRESH_RATE = None  # 最大刷新率  但是不一定可以达到

# 核心变量
game_graph = None
snake = []                         # 蛇的起始位置
food_x, food_y = None, None
aim_x, aim_y = None, None            # 蛇的起始方向
score = 0  # 分数

def square(x, y, size, sq_color):
    """绘制小正方形, 代表一格"""
    x, y = x*SIZE, y*SIZE
    turtle.penup()
    turtle.goto(x, y)
    turtle.color(sq_color)
    turtle.pendown()

    turtle.begin_fill()
    for i in range(4):
        turtle.forward(size)
        turtle.left(90)
    turtle.end_fill()


def frame():
    """绘制边框"""
    for i in range(LEFT - 1, RIGHT + 2, 1):
        square(i, BOTTOM - 1, SIZE, 'gray')
        square(i, TOP + 1, SIZE, 'gray')
    for i in range(BOTTOM - 1, TOP + 2, 1):
        square(LEFT - 1, i, SIZE, 'gray')
        square(RIGHT + 1, i, SIZE, 'gray')


def change(direction):
    """改变蛇的运动方向"""
    global aim_x, aim_y
    x, y = direction
    if (x != -aim_x or y != -aim_y) and (abs(x) + abs(y) == 1):
        aim_x = x
        aim_y = y


def inside(head_x, head_y):
    """判断蛇是否在边框内"""
    if LEFT <= head_x <= RIGHT and BOTTOM <= head_y <= TOP:
        return True
    else:
        return False

def new_food():
    """随机生成食物位置"""
    # todo 优化一下食物生成算法
    food = all_food.copy()    # 浅拷贝即可，内部使用的是元组
    for i in snake:            # 去掉蛇所在的位置
        food.remove(i)
    new_food_x, new_food_y = food.pop(random.randint(0, len(food) - 1))
    return new_food_x, new_food_y

def interpolate_color(start_color, end_color, steps):
    """生成两种颜色之间的渐变色"""
    for i in range(steps):
        r = start_color[0] + (end_color[0] - start_color[0]) * i / steps
        g = start_color[1] + (end_color[1] - start_color[1]) * i / steps
        b = start_color[2] + (end_color[2] - start_color[2]) * i / steps
        yield (r, g, b)

def draw_snake():
    """绘制蛇"""
    num_segments = len(snake)  # 渐变色的段数
    start_color = (1, 0.5, 0.5)  # 蛇尾
    end_color = (1, 0, 0)    # 蛇头
    for i, color in enumerate(interpolate_color(start_color, end_color, num_segments)):
        turtle.color(color)
        square(snake[i][0], snake[i][1], SIZE, color)
    # square(snake[-1][0], snake[-1][1], SIZE, 'red')  # 蛇头还得是红色

def move():
    global food_x, food_y, score

    change(pathfinding(game_graph))
    # 同步一下 game_graph 的状态
    game_graph.set_aim(aim_x, aim_y)
    game_graph.move_snake()

    head_move_x = snake[-1][0] + aim_x
    head_move_y = snake[-1][1] + aim_y

    # 判断是否撞到边框或者撞到自己
    # 此处也有修改，去除了原程序中对于尾部的错误判断
    if not inside(head_move_x, head_move_y) or ((head_move_x, head_move_y) in snake and (head_move_x,head_move_y) != snake[0]):
        square(head_move_x, head_move_y, SIZE, 'blue')
        turtle.update()
        print('得分: ', score)
        return

    snake.append((head_move_x, head_move_y))

    # 判断是否吃到食物以及是否胜利
    if head_move_x == food_x and head_move_y == food_y:
        score += 1  # 每吃到一个食物加1分
        if len(snake) >= len(all_food)*LOOSE:     # 限制松一点
            print('YOU WIN!')
            return
        else:
            food_x, food_y = new_food()
            # 更新食物同时要更新蛇蛇，更新在 game_graph.snake 里
            game_graph.set_food((food_x, food_y))
    else:
        snake.pop(0)

    turtle.clear()

    # 绘制边框, 蛇和食物
    frame()
    draw_snake()
    square(food_x, food_y, SIZE, 'green')

    # 显示分数
    turtle.penup()
    turtle.goto((LEFT-4) * SIZE, (TOP+4) * SIZE )
    turtle.color('black')
    turtle.write(f'Score: {score}', align='left', font=('Arial', 22, 'bold'))

    turtle.update()

    turtle.ontimer(move, int(1000 / MAX_REFRESH_RATE))  # 1 s = 1000 ms


def start(left=-6, right=6, top=6, bottom=-6, size=40, score_rate=3/4, max_fresh_rate=60):
    # 引入核心变量，用于赋值
    global game_graph
    global LEFT, RIGHT, TOP, BOTTOM
    global all_food, food_x, food_y
    global aim_x, aim_y
    global SIZE, MAX_REFRESH_RATE, LOOSE
    global snake, score

    # 常量赋值
    MAX_REFRESH_RATE = max_fresh_rate
    LOOSE = score_rate
    LEFT, RIGHT, TOP, BOTTOM = left, right, top, bottom
    SIZE = size
    for x in range(LEFT, RIGHT + 1, 1):
        for y in range(BOTTOM, TOP + 1, 1):
            all_food.append((x, y))

    # 核心变量赋值
    snake = [(0, 0), (1, 0)]   # list内一定要使用元组
    food_x, food_y = new_food()
    aim_x, aim_y = 0, 0
    score = len(snake)  # 初始化分数
    # 此处传入的是蛇蛇的深复制
    game_graph = GameGraph(copy.deepcopy(snake), (food_x, food_y),
                           {'xmin': left, 'xmax': right, 'ymin': bottom, 'ymax': top}, size)

    # Snake, 启动！
    # os.system("pause")
    turtle.setup((right - left + 10)*size, (top - bottom + 10)*size)
    turtle.title('贪吃蛇')
    turtle.hideturtle()
    turtle.tracer(False)   # 不显示轨迹
    turtle.delay(0)
    move()
    turtle.done()

if __name__ == '__main__':
    start()
