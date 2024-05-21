from RL_utils import get_train_decision, calculate_map_state, update_graph_state, generate_new_candy
from GameGraph import GameGraph
from model import RlModel
from Config import ConfigSet, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


class RLSnake:
    def __init__(self, config):
        # 设置🐍从地图中间开始，其中地图的信息保存在config这个disk中
        self.config = config
        self.start_point = config['start_point']
        self.graph = GameGraph(self.start_point, config['start_food'], config['edges'], config["size"])
        self.model = RlModel(config).to(device)
        self.choice = config['decision']
        self.gamma = config['gamma']
        self.bad_score = -10
        self.warn_score = -1
        self.correct_score = 1
        self.win_score = 10
        self.loss_function = nn.MSELoss()
        self.lr = config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.state_list = []  # 用于在训练中记录过程的state
        # 将config设置为如下的decision
        # self.decision = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def get_step_value(self, graph):
        # 根据目前的状态，计算各个choice所获得的相应的value
        # 几个标准
        # 第一，如果撞到边界或者蛇身体，则会产生一个负值
        # 第二，如果产生一个循环，那么会产生一个负值
        # 第三，如果吃到糖，产生一个正值
        # 第四，如果向糖走，产生一个较小的正值
        snake_head = graph.snake[-1]
        snake_tail = graph.snake[0]
        value_list = []
        for each in self.choice[graph.head_index]:
            next_station = (snake_head[0] + each[0], snake_head[1] + each[1])
            value = 0
            if not self.graph.is_valid_move(each):
                value += self.bad_score
            if next_station[0] == snake_tail[0] and next_station[1] == snake_tail[1]:  # 判断移动的方向刚好是蛇的尾部，增加一个小负项，防止出现循环
                value += self.warn_score
            if each[0] * (graph.food_x - snake_head[0]) > 0 or each[1] * (graph.food_y - snake_head[1]) > 0:
                value += self.correct_score
            if next_station[0] == graph.food_x and next_station[1] == graph.food_y:
                value += self.win_score
            value_list.append(value)
        return value_list

    def train_run(self, config, max_step=10000):
        # 设置训练的最大步数，防止出现循环
        # 首先尝试使用DQN
        # 每一轮开始初始化graph
        graph = GameGraph(snake=config['start_point'], food=config['start_food'],
                          edges=config['edges'], square_size=config["size"])
        graph = generate_new_candy(graph)
        for _ in range(max_step):
            self.optimizer.zero_grad()
            step_value = torch.tensor(self.get_step_value(graph), device=device)
            # 获取当前的graph的状态，并将其转化为一个一维向量
            graph_station = calculate_map_state(graph, choices=self.choice[graph.head_index]).view(-1).to(device)
            next_step_q_value = self.model(graph_station)
            _, max_id = torch.max(next_step_q_value, dim=-1)
            max_q = next_step_q_value[max_id]
            loss_list = []
            for j in range(len(step_value)):
                step_loss = self.loss_function(next_step_q_value[j], (step_value[j] + self.gamma * max_q))
                loss_list.append(step_loss)
            loss = sum(loss_list)
            loss.backward()
            self.optimizer.step()
            next_decision, decision_index = get_train_decision(config, graph.head_index,
                                                               next_step_q_value.detach().cpu())
            if graph.is_valid_move(next_decision):
                graph = update_graph_state(graph, decision_index, next_decision)
                continue
            else:
                self.state_list.append(len(graph.snake))
                break

    def run_test(self, graph=None):
        if graph is None:
            graph = GameGraph(self.start_point, self.config['start_food'], self.config['edges'], self.config["size"])
        model_state = torch.load(self.config['save_path'])
        self.model.load_state_dict(model_state)
        with torch.no_grad():
            while True:
                graph_station = torch.tensor(calculate_map_state(graph), device=device)
                next_step_q_value = self.model(graph_station)
                next_decision = self.choice[next_step_q_value.argmax(dim=-1)]
                graph, is_continue = update_graph_state(graph, next_decision)
                if not is_continue:
                    break


if __name__ == "__main__":
    config_ = ConfigSet(edges={'xmin': 0, 'xmax': 11, 'ymin': 0, 'ymax': 11},
                        start_point=[(6, 6)],
                        start_food=[1, 1],
                        size=24,
                        decision=[[(0, -1), (-1, 0), (1, 0)],
                                  [(1, 0), (0, -1), (0, 1)],
                                  [(0, 1), (1, 0), (-1, 0)],
                                  [(-1, 0), (0, 1), (0, -1)]],
                        gamma=0.8,
                        lr=0.0005,
                        input_layer=[11, 256],
                        hidden_layer=[256, 3],
                        save_path=r"G:/python_program/AI_snake/AI-snake-main/RL_model/model_param")
    RL = RLSnake(config_)
    for t in tqdm(range(100), total=100):
        RL.train_run(config_)
    torch.save(RL.model.state_dict(), r"G:\python_program\AI_snake\AI-snake-main\RL_model\model_param\model.pt")
    plt.plot([i for i in range(len(RL.state_list))], RL.state_list)
    plt.show()
    # 头的朝向分别为左，上，右，下, 移动方向分别为forward，turn left， turn right


