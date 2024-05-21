import random
import copy
from GameGraph import GameGraph
from queue import PriorityQueue


# 曼哈顿距离，如果使用这个启发式函数， 可能导致路线过于单一（完全追求最短）
def _heuristic1(start, end) -> float:
    return abs(end[0]-start[0]) + abs(end[1]-start[1])

# 欧氏距离，有着和上面一样的缺点
def _heuristic2(start, end) -> float:
    return ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5

# 曼哈顿距离 + 随机权重 (不需要找最短) --> 破圈的希望
def _heuristic3(start, end) -> float:
    manhattan_distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
    random_factor = random.uniform(0.9, 1.1)
    return manhattan_distance * random_factor
# 启发式函数
heuristic1 = _heuristic3
heuristic2 = None

def A_star(start, end, game_graph, func):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    in_close = dict()   # 是否在 close_list 的判断数组
    g_value_dict = dict()   # 记录 g_value 的字典
    parent = dict()    # 记录每个点的前驱
    q = PriorityQueue()    # 实现算法的优先队列， 同时也是 open_list

    q.put_nowait((func(start, end), start))    # (f(n), n)
    g_value_dict[start] = 0
    parent[start] = None

    while not q.empty():
        now = q.get_nowait()[1]
        in_close[now] = 1

        if now == end:
            break

        random.shuffle(moves)     # 随机是有必要的，防止蛇绕圈圈
        for move in moves:
            nxt = now[0] + move[0], now[1] + move[1]
            if game_graph.is_collision(nxt) or not game_graph.is_inside(nxt) or in_close.get(nxt, 0):
                continue

            new_g_value = g_value_dict[now] + 1

            if nxt not in g_value_dict or new_g_value < g_value_dict[nxt]:   # 不移除旧值，所以 (II) (III) 统一了
                g_value_dict[nxt] = new_g_value
                f_value = new_g_value + func(nxt, end)
                q.put_nowait((f_value, nxt))
                parent[nxt] = now

    # 寻找路径
    if end not in parent:    # 无可行路径
        return None
    path = []
    while end is not None:
        path.append(end)
        end = parent[end]
    path.reverse()
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
                stack.append((next_position, list(path)))   # 又是一个因为 py 没有指针而引发惨案， 直接传path有大问题

    return longest_path


def pathfinding(game_graph: GameGraph) -> (int, int):   # 传入的是 game_graph的引用，不会有额外的开销
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    food = (game_graph.food_x, game_graph.food_y)
    path_to_food = A_star(game_graph.snake[-1], food, game_graph, func=heuristic1)
    path_to_tail = A_star(game_graph.snake[-1], game_graph.snake[0], game_graph, func=heuristic1)
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
        path_to_tail2 = A_star(virtual_game_graph.snake[-1], virtual_game_graph.snake[0],
                               virtual_game_graph, func=heuristic1)
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


