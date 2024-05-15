import random
import copy
from GameGraph import GameGraph
from collections import deque

def bfs(start, end, game_graph: GameGraph):  # 最重要的函数之一
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # 使用广度优先搜索查找最短路径
    queue = deque()
    queue.append(start)

    parent = dict()     # 记录每个点的前驱
    parent[start] = None

    while queue:
        current = queue.popleft()

        if current == end:
            break

        random.shuffle(moves)   # 将moves随机，"破圈"的希望
        for move in moves:
            x, y = current[0] + move[0], current[1] + move[1]
            next_position = (x, y)
            if not game_graph.is_collision(next_position) and game_graph.is_inside(next_position):
                if next_position not in parent:
                    queue.append(next_position)
                    parent[next_position] = current

    if end not in parent:   # 没有找到路径
        return None

    path = []
    current = end    # 从末端开始寻找
    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()   # 翻转
    return path


def dfs_longest(start, end, game_graph: GameGraph):  # 最重要的函数之一
    visited_nodes = set()
    stack = [(start, [])]
    longest_path = None

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while stack:
        node, path = stack.pop()
        visited_nodes.add(node)
        path += [node]

        if node == end:
            if longest_path is None or len(path) > len(longest_path):
                longest_path = path

        # 此处不随机也很好
        random.shuffle(moves)
        for move in moves:
            x, y = node[0] + move[0], node[1] + move[1]
            next_position = (x, y)

            if (not game_graph.is_collision(next_position) and game_graph.is_inside(next_position)
                    and next_position not in visited_nodes):
                stack.append((next_position, list(path)))   # 又是一个因为 py 没有指针而引发惨案

    return longest_path


def pathfinding(game_graph: GameGraph) -> (int, int):   # 传入的是 game_graph的引用，不会有额外的开销
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    food = (game_graph.food_x, game_graph.food_y)
    path_to_food = bfs(game_graph.snake[-1], food, game_graph)
    path_to_tail = bfs(game_graph.snake[-1], game_graph.snake[0], game_graph)
    # 只剩最后一个食物的特判
    if len(game_graph.snake) == game_graph.graph_size - 1 and path_to_food is not None:
        return path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1]

    # 长度为1 特判
    if path_to_food is not None and len(game_graph.snake) == 1:
        return path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1]   # 直接跑向食物就可以

    # 能够吃到食物
    if path_to_food is not None:
        virtual_game_graph = copy.deepcopy(game_graph)
        move = (path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1])
        virtual_game_graph.aim_x = move[0]
        virtual_game_graph.aim_y = move[1]
        virtual_game_graph.move_snake()  # 得到虚拟蛇
        path_to_tail2 = bfs(virtual_game_graph.snake[-1], virtual_game_graph.snake[0], virtual_game_graph)
        # 虚拟蛇可以到达尾巴，则原蛇可以移动
        if path_to_tail2 is not None:
            return move
        # 虚拟蛇无法到达尾巴，但是原蛇可以到达尾巴，向 到尾巴的最长路径前进
        elif path_to_tail is not None:
            longest_path_to_tail = dfs_longest(game_graph.snake[-1], game_graph.snake[0], game_graph)   # 最短路存在，那么最长路也一定存在
            move = (longest_path_to_tail[1][0] - longest_path_to_tail[0][0],
                    longest_path_to_tail[1][1] - longest_path_to_tail[0][1])
            return move

    # 不能吃到食物且原蛇可以到达尾巴，向远离尾巴的方向前进
    if path_to_food is None and path_to_tail is not None:
        longest_path_to_tail = dfs_longest(game_graph.snake[-1], game_graph.snake[0], game_graph)
        move = (longest_path_to_tail[1][0] - longest_path_to_tail[0][0],
                longest_path_to_tail[1][1] - longest_path_to_tail[0][1])
        return move

    # Wander 来点随机性
    right_moves = []
    for move in moves:
        if game_graph.is_valid_move(move):
            right_moves.append(move)
    if len(right_moves) == 0:
        return 0, 1      # 这蛇直接撞死吧，没救了
    return right_moves[random.randint(0, len(right_moves)-1)]   # 两端都包括

    # todo 当没法吃到食物并且没法找到尾巴时，采用 A* 算法向远离食物的方向前进


# 测试代码
if __name__ == '__main__':
    game = GameGraph([(0, 0), (0, 1)], (0, 0), {'xmin': -20, 'xmax': 18, 'ymin': -19, 'ymax': 19}, 10)
    print(dfs_longest((0, 0), (1, 1), game))
