import argparse
import snake
import importlib
from ga import GA
import GameGraph
from nn import Net
import torch
import numpy as np
'''
AI_Snake 完全使用指南  =w=. 
1. 新建pathfinding_XX 模块  (模块名没有强制要求)
2. 该模块内完成pathfinding函数 (函数名强制要求) ，该函数接受一个GameGraph对象,返回一个元组(int, int)用于指定方向
3. 命令行中调用 'python main.py --func pathfinding_XX' (你也可以加上其他自定义函数) (建议写个脚本，或者使用pycharm的启动项参数来实现自动测试)
'''

def main():
    parser = argparse.ArgumentParser(description='Snake game with command line arguments')
    parser.add_argument('-l', '--left', type=int, default=-4, help='left boundary')
    parser.add_argument('-r', '--right', type=int, default=4, help='right boundary')
    parser.add_argument('-t', '--top', type=int, default=4, help='top boundary')
    parser.add_argument('-b', '--bottom', type=int, default=-4, help='bottom boundary')
    parser.add_argument("--size", type=int, default=40, help='每一小格的边长')
    parser.add_argument("--rate", type=int, default=60, help='最大刷新率')
    parser.add_argument("--score", type=float, default=11/12, help="获胜所需的分数占比")
    parser.add_argument('-f', "--func", type=str, default="pathfinding_MLP", help="寻路函数所在文件(模块)")
    parser.add_argument('-i', '--inherit', action="store_true", help="Whether to load genes from path ./genes/all.")
    parser.add_argument('-s', '--show', action="store_true", help='whether to show the best individual to play snake after each envolve.')
    args = parser.parse_args()

    # 将包含寻路函数的模块导入，并且获取 pathfinding 函数
    module = importlib.import_module(args.func)
    pathfinding_func = getattr(module, "pathfinding")

    # 检查获取的对象是否为函数
    if not callable(pathfinding_func):
        raise ValueError(f"Object {args.func} is not a function")

    # snake.start(args.left, args.right, args.top, args.bottom, args.size, args.score, args.rate, pathfinding_func)

    ''' 遗传算法 '''
    ga = GA()

    if args.inherit:
        ga.inherit_ancestor()
    else:
        ga.generate_ancestor()


    generation = 0
    record = 0
    while True:
        generation += 1
        ga.evolve()
        print(
            f"generation: {generation}, record: {record}, best score: {ga.best_individual.score}, average score: {ga.avg_score}")

        if ga.best_individual.score >= record:
            record = ga.best_individual.score
            # ga.save_best()

        if args.show:
            # Assuming `start` method initiates the game with given game graph and pathfinding function
            game_graph = GameGraph(show=True, genes=ga.best_individual.genes)
            snake.start(args.left, args.right, args.top, args.bottom, args.size, args.score, args.rate, pathfinding_func)


        # if generation % 20 == 0:
        #     ga.save_all()

if __name__ == '__main__':
    main()

