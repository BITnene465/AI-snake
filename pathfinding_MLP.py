import random
from ga_train import Individual
import torch
import copy
from typing import Tuple
from GameGraph import GameGraph
from nn import Net



# 模型初始化和权重加载
weights_path = 'genes/best_genes_gen3000.txt'
with open(weights_path, 'r') as file:
    weights = file.read().split()
    weights = [float(w) for w in weights]
    model = Net(n_input=Individual.n_input, n_hidden1=Individual.n_hidden1,
                       n_hidden2=Individual.n_hidden2, n_output=Individual.n_output, weights=weights)
    model.eval()      # 无需计算梯度

def pathfinding(game_graph: GameGraph) -> Tuple[int, int]:
    input_vector = game_graph.to_input_vector2()
    return _dirMapping(game_graph.GetAim(), model.predict(input_vector))

def _dirMapping(raw_direction: Tuple[int, int], label: int) -> Tuple[int, int]:
    """
    label : 0, 1, 2 分别表示蛇的左转，保持，右转操作
    raw_direction : 蛇的原朝向
    """
    if label == 1:
        return copy.copy(raw_direction)
    elif label == 0:
        return raw_direction[1], raw_direction[0]
    elif label == 2:
        return -raw_direction[1], -raw_direction[0]
    else:
        print("模型输出类型不匹配")
