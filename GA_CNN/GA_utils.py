import sys
sys.path.append('../')  # 添加上一级目录到 Python 解释器搜索路径中

from GameGraph import GameGraph
import snake
from mlp_model import MLP, get_mpl_model
import torch
import random

# 辅助函数
def model2gene(model: MLP):
    # 获取卷积层和全连接层的权重
    fc1_weight = model.fc1.weight
    fc2_weight = model.fc2.weight
    fc3_weight = model.fc3.weight
    # 展平并连接权重作为遗传算法的gene序列
    gene_seq = torch.cat((fc1_weight.view(-1), fc2_weight.view(-1), fc3_weight.view(-1)))
    return gene_seq


def gene2model(gene_seq, input_size=24, hidden_size1=128, hidden_size2=32, output_size=3):
    # 分割基因序列以获得每个全连接层的权重参数
    fc1_weight_len = input_size*hidden_size1
    fc2_weight_len = hidden_size1*hidden_size2
    fc3_weight_len = hidden_size2*output_size

    fc1_weight = gene_seq[: fc1_weight_len]
    fc2_weight = gene_seq[fc1_weight_len: fc1_weight_len + fc2_weight_len]
    fc3_weight = gene_seq[fc1_weight_len + fc2_weight_len: ]

    # 构建模型
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)

    # 将解码的权重参数设置到模型中
    model.fc1.weight.data = fc1_weight
    model.fc2.weight.data = fc2_weight
    model.fc3.weight.data = fc3_weight

    return model

# 游戏方面
def getState(gamegraph: GameGraph):
    pass

def decisionMapping(aim: (int, int), rel: int) -> (int, int):
    """
    rel : 0(保持), 1(左转), 2(右转)  三种label， 由MLP得到
    aim : (1, 0), (-1, 0), (0, 1), (0, -1)  四种朝向
    """
    if rel == 0:
        return aim
    elif rel == 1:
        return aim[1], aim[0]
    elif rel == 2:
        return -aim[1], -aim[0]


def simulate_game(gene_seq, left, right, top, bottom) -> float:
    snake_brain = gene2model(gene_seq)
    snake_brain.eval()
    score = 0
    def _pathfinding(_gamegraph: GameGraph):
        nonlocal snake_brain, score
        rel = snake_brain(getState(_gamegraph))
        score = len(_gamegraph.snake)
        return decisionMapping(_gamegraph.GetAim(), rel)

    snake.start(left=left, right=right, top=top, bottom=bottom,
                score_rate=1, pathfinding_func=_pathfinding, is_train=True)

    return score

# GA方面
def initialize_population(population_size, gene_length):
    population = []
    for _ in range(population_size):
        gene = torch.randn(gene_length)
        population.append(gene)
    return population

# 计算适应度
def compute_fitness(gene, fitness_function):
    return fitness_function(gene)

# 选择父母
def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        # 使用轮盘赌选择方法
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current > pick:
                parents.append(population[i])
                break
    return parents

# 交叉操作
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1_index = random.randint(0, len(parents) - 1)
        parent2_index = random.randint(0, len(parents) - 1)
        parent1 = parents[parent1_index]
        parent2 = parents[parent2_index]
        crossover_point = random.randint(0, len(parent1) - 1)
        child = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
    return offspring

# 变异操作
def mutate(offspring, mutation_rate):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(offspring[i]) - 1)
            offspring[i][mutation_point] += torch.randn(1)
    return offspring


if __name__ == '__main__':
    gene_seq = torch.randn(24*128+128*32+32*3)
    simulate_game(gene_seq, -10, 10, 10, -10)