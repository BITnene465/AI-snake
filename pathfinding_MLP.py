import random
from ga_train import Individual
import torch
from GameGraph import GameGraph
from nn import Net

def pathfinding(game_graph):
    # 将游戏状态转换为神经网络的输入向量
    input_vector = game_graph.to_input_vector()

    # 模型初始化和权重加载
    weights_path = 'best_genes_1.txt'
    with open(weights_path, 'r') as file:
        weights = file.read().split()
        weights = [float(w) for w in weights]
        model = Net(n_input=Individual.n_input, n_hidden1=Individual.n_hidden1,
                           n_hidden2=Individual.n_hidden2, n_output=Individual.n_output, weights=weights)

    # todo 更改以下代码
    # 使用模型预测输出向量
    output_vector = model(torch.tensor(input_vector).float())

    # 计算当前方向以避免蛇反向移动
    current_direction = (game_graph.snake[-1][0] - game_graph.snake[-2][0],
                         game_graph.snake[-1][1] - game_graph.snake[-2][1])
    opposite_direction = (-current_direction[0], -current_direction[1])

    # 按概率降序排列输出方向
    sorted_indices = output_vector.argsort(descending=True)

    # 遍历排序后的方向索引，找出非反向的合法移动方向
    for idx in sorted_indices:
        direction = GameGraph.DIRECTIONS[idx.item()]
        if direction != opposite_direction:
            return idx.item()

    # 如果所有方向均为反向，则默认返回概率最高的方向
    return sorted_indices[0].item()
