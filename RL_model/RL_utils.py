from model import RlModel
from Config import ConfigSet, set_seed
import random
import torch
from GameGraph import GameGraph
set_seed(42)


def get_step_thread(next_step_value: torch.Tensor):  # 设置每一步操作的区间值
    next_step_list = next_step_value
    next_step_list = torch.softmax(next_step_list, dim=-1)
    next_step_info = []
    for i in range(len(next_step_list)):
        if i == 0:
            next_step_info.append(next_step_list[i])
        else:
            next_step_info.append(next_step_list[i] + next_step_info[-1])
    return next_step_info


def generate_decision(next_step_info):  # 在按照每一步相关的值进行操作时，生成相关的决策
    _x = torch.rand((1,))
    for i in range(len(next_step_info)):
        if _x < next_step_info[i]:
            return i


def get_train_decision(config, head_index, next_step_value: torch.Tensor) -> [(int, int)]:
    # 在训练时使用这个函数生成下一步的决策，90%的概率按照概率进行，10%的概率进行随机行走
    # 这里将每一步获得的value进行softmax操作，然后生成一个随机数，当随机数在某个区间中时，就采取哪个行为
    # 这样可以解决一个问题，就是当多个决策能够获得相同的value时，应该怎么走
    # 不管如何，直接按照贪心进行选择

    decision = config['decision'][head_index]
    if torch.rand((1,)) < 1:
        # next_step_info = get_step_thread(next_step_value)
        # next_step = generate_decision(next_step_info)
        next_step = next_step_value.argmax(dim=-1)
        return decision[next_step], next_step
    else:
        index = torch.randint(0, len(next_step_value) - 1, (1,))
        return decision[index], index


def calculate_map_state(graph_: GameGraph, choices):
    # 将图的状态转化为11个坐标，分别代表着(三个行为是否是danger的，当前头的移动方向，食物所在方向)
    # 其中，snake是蛇蛇的坐标，edges是包含地图边界信息的一个字典
    # target是躺过的坐标，在每次吃到糖果时会进行更新
    snake = graph_.snake
    edges = graph_.edges
    target = (graph_.food_x, graph_.food_y)
    x_ = edges['xmax'] - edges['xmin'] + 1
    y_ = edges['ymax'] - edges['ymin'] + 1
    graph_map = torch.zeros((x_, y_))
    state_dict = []
    for each in choices:
        state_dict.append(graph_.is_valid_move(each) + 0)
    direct_list = [0, 0, 0, 0]
    direct_list[graph_.head_index] = 1
    state_dict.extend(direct_list)
    # 计算像哪个方向前进能够获得糖果
    x_d = graph_.food_x - snake[-1][0]
    y_d = graph_.food_y - snake[-1][1]
    if graph_.head_index == 0:
        state_dict.extend([(y_d < 0) + 0, (y_d > 0) + 0, (x_d < 0) + 0, (x_d > 0) + 0])
    elif graph_.head_index == 1:
        state_dict.extend([(x_d > 0) + 0, (x_d < 0) + 0, (y_d > 0) + 0, (y_d < 0) + 0])
    elif graph_.head_index == 2:
        state_dict.extend([(y_d > 0) + 0, (y_d < 0) + 0, (x_d > 0) + 0, (x_d < 0) + 0])
    elif graph_.head_index == 3:
        state_dict.extend([(x_d < 0) + 0, (x_d > 0) + 0, (y_d < 0) + 0, (y_d > 0) + 0])
    return torch.tensor(state_dict, dtype=torch.float32)


def generate_new_candy(graph):
    edges = graph.edges
    x_ = edges['xmax'] - edges['xmin'] + 1
    y_ = edges['ymax'] - edges['ymin'] + 1
    target_index = []
    for i in range(x_):
        for j in range(y_):
            if (i, j) not in graph.snake:
                target_index.append((i, j))
    candy_index = torch.randint(0, len(target_index), (1,))
    graph.food_x = target_index[candy_index][0]
    graph.food_y = target_index[candy_index][1]
    return graph


def update_graph_state(graph_: GameGraph, decision_index, decision_: [(int, int)]) -> (GameGraph, bool):
    # 这里的snake是一个列表，通过pop其第0个元素，在末尾添加一个新的元素来模拟一个队列
    # 这个函数的返回值为更新后的graph
    snake = graph_.snake
    snake_head = snake[-1]
    next_snake_head = (snake_head[0] + decision_[0], snake_head[1] + decision_[1])
    if decision_index == 0:
        graph_.head_index = graph_.head_index
    elif decision_index == 1:
        graph_.head_index = int((graph_.head_index - 1) % 4)
    elif decision_index == 2:
        graph_.head_index = int((graph_.head_index + 1) % 4)
    if next_snake_head[0] == graph_.food_x and next_snake_head[1] == graph_.food_y:  # 假如吃到糖，则更新糖的位置
        snake.append(next_snake_head)
        graph_.snake = snake
        graph_ = generate_new_candy(graph_)

    else:
        snake.pop(0)
        snake.append(next_snake_head)
        graph_.snake = snake

    return graph_


