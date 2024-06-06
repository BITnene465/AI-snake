import sys

sys.path.append("./")

import matplotlib.pyplot as plt
from ga_train import Individual
from nn import Net
import GameGraph
import copy
from typing import Tuple
from snake import SnakeGame
import json

weights_host = '../genes/best_genes_gen'


def EvalScore(model):
    game = SnakeGame(not_display_on_gui=True)
    game.setup_game()
    gg: GameGraph = game.getGamegraph()
    life_time = 100
    steps = 0  # 记录运行的总步数
    score = gg.GetScore()  # 初始化分数
    while True:
        life_time -= 1
        steps += 1
        input_vector = gg.to_input_vector2()
        direction = _dirMapping(gg.GetAim(), model.predict(input_vector))
        res, gg = game.move_StepByStep(direction)

        if score < gg.GetScore():  # 当蛇吃到食物后，增加寿命
            score = gg.GetScore()
            life_time += 100
            life_time = min(life_time, 400)
        if not res or life_time <= 0:  # 这里可以调整终止条件，比如达到一定分数或步数
            break
    return score


def getModel(weights_path):
    with open(weights_path, 'r') as file:
        weights = file.read().split()
        weights = [float(w) for w in weights]
        model = Net(n_input=Individual.n_input, n_hidden1=Individual.n_hidden1,
                    n_hidden2=Individual.n_hidden2, n_output=Individual.n_output, weights=weights)
        model.eval()  # 无需计算梯度
    return model


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


def save_data(start, end, interval, repeat=10, data_path="./scores.json"):
    data = {"generations": [], "scores": []}

    for gen in range(start, end, interval):
        weights_path = weights_host + str(gen) + '.txt'
        model = getModel(weights_path)
        score = 0
        for _ in range(repeat):
            score += EvalScore(model)
        score /= repeat

        print(f"gen{gen}, score:{score}")
        data["generations"].append(gen)
        data["scores"].append(score)

    with open(data_path, 'w') as f:
        json.dump(data, f)


def plot_data(data_path="./scores.json", save_path="./performance_plot.png"):
    with open(data_path, 'r') as f:
        data = json.load(f)

    generations = data["generations"]
    scores = data["scores"]

    # 画图部分
    plt.figure(figsize=(15, 5))
    plt.plot(generations, scores, marker='o', linestyle='-', color='b', markersize=5)
    plt.xlabel('Generation')
    plt.ylabel('Score (best individual)')
    plt.title('Evolution of Snake AI Performance over Generations')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    save_data(0, 6500, 25, repeat=25, data_path="./scores.json")
    plot_data(data_path="./scores.json", save_path="./performance_plot.png")
