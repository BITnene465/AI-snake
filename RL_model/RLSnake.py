from RL_utils import get_train_decision, calculate_map_state, update_graph_state
from GameGraph import GameGraph
from model import RlModel
from Config import ConfigSet, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
set_seed(42)


class RLSnake:
    def __init__(self, config):
        # 设置🐍从地图中间开始，其中地图的信息保存在config这个disk中
        self.config = config
        self.start_point = config['start_point']
        self.graph = GameGraph(self.start_point, config['start_food'], config['edges'], config["size"])
        self.model = RlModel(config)
        self.choice = config['decision']
        self.bad_score = -10
        self.warn_score = -1
        self.correct_score = 1
        self.win_score = 10
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_step_value(self):
        # 根据目前的状态，计算各个choice所获得的相应的value
        # 几个标准
        # 第一，如果撞到边界或者蛇身体，则会产生一个负值
        # 第二，如果产生一个循环，那么会产生一个负值
        # 第三，如果吃到糖，产生一个正值
        # 第四，如果向糖走，产生一个较小的正值
        snake_head = self.graph.snake[-1]
        snake_tail = self.graph.snake[0]
        value_list = []
        for each in self.choice:
            next_station = (snake_head[0] + each[0], snake_head[1] + each[1])
            value = 0
            if not self.graph.is_valid_move(next_station):
                value += self.bad_score
            if next_station[0] == snake_tail[0] and next_station[1] == snake_tail[1]:  # 判断移动的方向刚好是蛇的尾部，增加一个小负项，防止出现循环
                value += self.warn_score
            if each[0] * (self.graph.food_x - snake_head[0]) > 0 or each[1] * (self.graph.food_y - snake_head[1]) > 0:
                value += self.correct_score
            if next_station[0] == self.graph.food_x and next_station[1] == self.graph.food_y:
                value += self.win_score
            value_list.append(value)
        return value_list

    def train_run(self, max_step=10000):
        # 设置训练的最大步数，防止出现循环
        for i in tqdm(range(max_step), total=max_step):
            step_value = self.get_step_value()
            next_decision = get_train_decision(self.config, step_value)

            # 中间训练模型

            self.graph = update_graph_state(self.graph, next_decision)



