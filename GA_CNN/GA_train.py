from GA_utils import *

# 游戏超参数设置
left=-10
right=10
top=10
bottom=-10   # 共有 441 个方格

# 模型超参数设置
device = 'cpu'
input_size = 24
hidden_size1 = 128
hidden_size2 = 32
output_size = 3   # 控制左转，右转，保持前进

# 遗传算法超参数设置
population_size = 1000
gene_length = 24*128 + 128*32 + 32*3
num_generations = 100
num_parents = 5
crossover_rate = 0.8
mutation_rate = 0.1

# 可选择适应性函数
def fitness1(gene):




# 遗传算法实现
def genetic_algorithm(population_size: int, gene_length: int, num_generations: int,
                      num_parents: int, crossover_rate: float, mutation_rate: float, fitness_func=fitness1):
    # 初始化种群
    population = initialize_population(population_size, gene_length)

    for generation in range(num_generations):
        # 计算种群中每个个体的适应度
        fitness_scores = [compute_fitness(gene, fitness_func) for gene in population]

        # 选择父母
        parents = select_parents(population, fitness_scores, num_parents)

        # 交叉操作
        num_offspring = population_size - num_parents
        offspring = crossover(parents, num_offspring)

        # 变异操作
        offspring = mutate(offspring, mutation_rate)

        # 更新种群
        population = parents + offspring

        # 输出每一代的最优个体及其适应度
        best_index = fitness_scores.index(max(fitness_scores))
        best_gene = population[best_index]
        best_fitness = fitness_scores[best_index]
        print(f"Generation {generation + 1}: Best Fitness: {best_fitness}, Best Gene: {best_gene}")


# 运行遗传算法
genetic_algorithm(population_size, gene_length, num_generations, num_parents, crossover_rate, mutation_rate)