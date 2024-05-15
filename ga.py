import random
import argparse
import numpy as np
from snake import start
import os
from nn import Net
import pathfinding_MLP
from GameGraph import GameGraph

score = 0
snake = []


class Individual:

    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.seed = None

    def get_fitness(self):
        global score, snake  # 确保score和snake是全局变量

        # 生成并设置神经网络的权重
        nn_weights = np.array(self.genes)
        model = Net(32, 20, 12, 4, nn_weights)  # 创建神经网络实例

        # 定义pathfinding函数
        pathfinding = pathfinding_MLP.pathfinding

        # 初始化游戏并传递寻路函数
        start(left=-4, right=4, top=4, bottom=-4, size=40, score_rate=11 / 12, max_fresh_rate=60,
              pathfinding_func=pathfinding)

        # 评估结果
        self.score = score
        self.steps = len(snake)
        self.fitness = (self.score + 1 / max(1, self.steps)) * 100000  # 确保分母不为零


class GA:
    """Genetic Algorithm.
    Attributes:
        p_size: Size of the parent generation.
        c_size: Size of the child generation.
        genes_len: Length of the genes.
        mutate_rate: Probability of the mutation.
        population: A list of individuals.
        best_individual: Individual with best fitness.
        avg_score: Average score of the population.
        verbose: Enable detailed logging.
    """

    def __init__(self, p_size=100, c_size=400, genes_len=32*20+20*12+12*4+20+12+4, mutate_rate=0.1):
        """
        Initializes the genetic algorithm with given parameters.

        Args:
            p_size (int): Number of parents in the population, i.e., the size of the population.
            c_size (int): Number of children to produce each generation.
            genes_len (int): Number of genes per individual, should match the neural network's expected input size.
            mutate_rate (float): Mutation rate, likelihood of a gene undergoing mutation.
        """
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))

    # def inherit_ancestor(self):
    #     """Load genes from './genes/all/{i}', i: the ith individual."""
    #     for i in range(self.p_size):
    #         pth = os.path.join("genes", "all", str(i))
    #         with open(pth, "r") as f:
    #             genes = np.array(list(map(float, f.read().split())))
    #             self.population.append(Individual(genes))

    def crossover(self, c1_genes, c2_genes):
        """Single point crossover."""
        point = np.random.randint(0, self.genes_len)
        c1_genes[:point + 1], c2_genes[:point + 1] = c2_genes[:point + 1], c1_genes[:point + 1]

    def mutate(self, c_genes):
        """Gaussian mutation with scale of 0.2."""
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]

    def elitism_selection(self, size):
        """Select the top #size individuals to be parents."""
        population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def roulette_wheel_selection(self, size):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        return selection

    def evolve(self):
        '''The main procss of Genetic Algorithm.'''
        sum_score = 0
        for individual in self.population:
            individual.get_fitness()
            sum_score += individual.score
        self.avg_score = sum_score / len(self.population)

        self.population = self.elitism_selection(self.p_size)  # Select parents to generate children.
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        # Generate children.
        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)

    # def save_best(self):
    #     """Save the best individual that can get score so far."""
    #     score = self.best_individual.score
    #     genes_pth = os.path.join("genes", "best", str(score))
    #     with open(genes_pth, "w") as f:
    #         for gene in self.best_individual.genes:
    #             f.write(str(gene) + " ")
    #     seed_pth = os.path.join("seed", str(score))
    #     with open(seed_pth, "w") as f:
    #         f.write(str(self.best_individual.seed))
    #
    # def save_all(self):
    #     """Save the population."""
    #     for individual in self.population:
    #         individual.get_fitness()
    #     population = self.elitism_selection(self.p_size)
    #     for i in range(len(population)):
    #         pth = os.path.join("genes", "all", str(i))
    #         with open(pth, "w") as f:
    #             for gene in self.population[i].genes:
    #                 f.write(str(gene) + " ")