import torch
from nn import Net  # Ensure your neural network class is correctly imported
import random
import numpy as np

# Instantiate the neural network
weights = [random.random() for i in range(32 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]
model = Net(32, 20, 12, 4, weights)
model.eval()  # Set the model to evaluation mode if not already set


def get_input_vector(game_graph):
    head_x, head_y = game_graph.snake[-1]
    neck_x, neck_y = game_graph.snake[-2]
    tail_x, tail_y = game_graph.snake[0]
    tail_dir_x, tail_dir_y = game_graph.snake[1]

    # 初始化输入列表
    inputs = []

    # 1. 蛇首方向
    head_direction = (head_x - neck_x, head_y - neck_y)
    head_direction_vector = [0, 0, 0, 0]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上，下，右，左
    if head_direction in directions:
        head_direction_vector[directions.index(head_direction)] = 1
    inputs.extend(head_direction_vector)

    # 2. 蛇尾方向
    tail_direction = (tail_x - tail_dir_x, tail_y - tail_dir_y)
    tail_direction_vector = [0, 0, 0, 0]
    if tail_direction in directions:
        tail_direction_vector[directions.index(tail_direction)] = 1
    inputs.extend(tail_direction_vector)

    # 3. 蛇首八个方向上是否有食物，是否有自身，与墙的距离
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    food_presence = [0] * 8
    body_presence = [0] * 8
    wall_distances = [0] * 8

    for i, (dir_x, dir_y) in enumerate(directions):
        distance = 1
        while True:
            check_x = head_x + dir_x * distance
            check_y = head_y + dir_y * distance

            if not game_graph.is_inside((check_x, check_y)):
                wall_distances[i] = distance
                break

            if (check_x, check_y) in game_graph.snake:
                body_presence[i] = 1
                break

            if (check_x, check_y) == (game_graph.food_x, game_graph.food_y):
                food_presence[i] = 1

            distance += 1

    inputs.extend(food_presence)
    inputs.extend(body_presence)
    inputs.extend(wall_distances)


    return torch.tensor([inputs], dtype=torch.float32)

def pathfinding(game_graph):
    input_tensor = get_input_vector(game_graph)
    with torch.no_grad():
        outputs = model(input_tensor)
        move_idx = torch.argmax(outputs).item()
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    move = directions[move_idx]
    return move
