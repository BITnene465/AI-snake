import copy
import time

import numpy as np
import random
import os

import torch.cuda

from nn import Net
from snake import SnakeGame  # 确保这个模块可以正常导入
from typing import Tuple
from GameGraph import GameGraph  # 确保这个模块可以正常导入

p_size = 300
c_size = 200
mutate_rate = 0.10

class Individual:
    # 类变量
    n_input = 32
    n_hidden1 = 32
    n_hidden2 = 24
    n_output = 3
    genes_len = (n_input * n_hidden1 + n_hidden1 * n_hidden2 + n_hidden2 * n_output
                 + n_hidden1 + n_hidden2 + n_output)
    def __init__(self, genes, device):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.device = device        #  cpu 训练更快
        self.network = Net(n_input=Individual.n_input, n_hidden1=Individual.n_hidden1,
                           n_hidden2=Individual.n_hidden2, n_output=Individual.n_output, weights=self.genes,
                           device=self.device)
        self.network.eval()    # 不需要计算梯度

    def get_fitness(self):
        game = SnakeGame(not_display_on_gui=True)
        game.setup_game()
        gg = game.getGamegraph()
        life_time = 100
        steps = 0    # 记录运行的总步数
        self.score = gg.GetScore()   # 初始化分数
        self.fitness = 0
        while True:
            dis1 = gg.getDis()

            life_time -= 1
            steps += 1
            input_vector = gg.to_input_vector2()
            direction = self._dirMapping(gg.GetAim(), self.network.predict(input_vector))
            res, gg = game.move_StepByStep(direction)

            # dis2 = gg.getDis()
            # if dis2 < dis1:
            #     self.fitness += 1.5
            # else:
            #     self.fitness -= 1
            if self.score < gg.GetScore():     # 当蛇吃到食物后，增加寿命
                self.score = gg.GetScore()
                life_time += 100
                life_time = min(life_time, 400)
            if not res or life_time <= 0:  # 这里可以调整终止条件，比如达到一定分数或步数
                break
        self.fitness += self.score

    def _dirMapping(self, raw_direction: Tuple[int, int], label: int) -> Tuple[int, int]:
        """
        辅助函数，私有方法
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


# GA类和其它方法根据实际需求进行修改


class GA:
    def __init__(self, p_size, c_size, genes_len, mutate_rate):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0
        self.device = 'cpu'
    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes, self.device))

    def inherit_ancestor(self):
        for i in range(self.p_size):
            pth = os.path.join("genes", "genes/all", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(Individual(genes, self.device))

    def crossover(self, c1_genes, c2_genes):
        point = np.random.randint(0, self.genes_len)
        c1_genes[:point + 1], c2_genes[:point + 1] = c2_genes[:point + 1], c1_genes[:point + 1]

    def mutate(self, c_genes):
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]

    def elitism_selection(self, size):
        population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def roulette_wheel_selection(self, size):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        if wheel == 0:
            raise ValueError("Total fitness is zero, selection cannot proceed.")

        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        if len(selection) < size:
            raise ValueError(f"Not enough individuals selected: expected {size}, got {len(selection)}")

        return selection

    def calc_popu_fitness(self):
        for individual in self.population:
            individual.get_fitness()

    def evolve(self):
        sum_score = 0
        for individual in self.population:
            sum_score += individual.score
        self.avg_score = sum_score / len(self.population)

        self.population = self.elitism_selection(self.p_size)
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1, c2 = Individual(c1_genes, self.device), Individual(c2_genes, self.device)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children[:self.c_size])

    def save_best(self, generation, filename="best_genes.txt"):
        if self.best_individual is not None:
            filename_with_gen = f"{filename.split('.')[0]}_gen{generation}.txt"
            full_path = os.path.join("genes", filename_with_gen)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as file:
                genes_str = ' '.join(map(str, self.best_individual.genes))
                file.write(genes_str)
            print(f"Saved best genes to {full_path}")
        else:
            print("No best individual to save.")

    def save_all(self):
        population = self.elitism_selection(self.p_size)
        for i in range(len(population)):
            pth = os.path.join("genes", "genes/all", str(i))
            with open(pth, "w") as f:
                for gene in self.population[i].genes:
                    f.write(str(gene) + " ")

    def train(self, generations, save_interval=100):
        for generation in range(generations):
            stt = time.time()

            self.calc_popu_fitness()
            self.avg_score = sum(ind.score for ind in self.population) / len(self.population)
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            self.best_individual = self.population[0]

            edt = time.time()
            print(f"Generation {generation + 1}: Best Score = {self.best_individual.score}, Avg Score = {self.avg_score:.3f}, time: {edt-stt:.3f}")
            if generation % save_interval == 0 or generation == generations - 1:
                self.save_best(generation)

            self.evolve()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("device: cuda is available")
    else:
        print("device: cuda is not available")
    print("训练开始".center(100, '='))
    ga = GA(p_size, c_size, Individual.genes_len, mutate_rate)
    print("当前设备: ", ga.device)
    ga.generate_ancestor()
    ga.train(20000, save_interval=25)
