import random
import copy
from GameGraph import GameGraph
from queue import PriorityQueue
from snake import SnakeGame

# 曼哈顿距离，如果使用这个启发式函数， 可能导致路线过于单一（完全追求最短）
def heuristic1(start, end) -> float:
    return abs(end[0]-start[0]) + abs(end[1]-start[1])

def A_star(start, end, game_graph, h_func):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    in_close = dict()   # 是否在 close_list 的判断数组
    g_value_dict = dict()   # 记录 g_value 的字典
    parent = dict()    # 记录每个点的前驱
    q = PriorityQueue()    # 实现算法的优先队列， 同时也是 open_list

    q.put_nowait((h_func(start, end), start))    # (f(n), n)
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
                f_value = new_g_value + h_func(nxt, end)
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

def longest_to_(end, game_graph: GameGraph):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random.shuffle(moves)
    head = game_graph.snake[-1]
    mmove = None
    mvalue = None
    for move in moves:
        now_head = head[0] + move[0], head[1] + move[1]
        if move[0] == -game_graph.aim_x and move[1] == -game_graph.aim_y or not game_graph.is_valid_move(move):
            continue
        virtual_gg = copy.deepcopy(game_graph)
        virtual_gg.set_aim(move[0], move[1])
        virtual_gg.move_snake()
        if A_star(virtual_gg.snake[-1], virtual_gg.snake[0], virtual_gg, heuristic1) is None:
            continue
        if mvalue is None or mvalue <= heuristic1(now_head, end):
            mmove = move
            mvalue = heuristic1(now_head, end)
    return mmove


def pathfinding(game_graph: GameGraph) -> (int, int):   # 传入的是 game_graph的引用，不会有额外的开销
    tail = game_graph.snake[0]
    food = game_graph.GetFood()
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    path_to_food = A_star(game_graph.snake[-1], food, game_graph, h_func=heuristic1)
    path_to_tail = A_star(game_graph.snake[-1], game_graph.snake[0], game_graph, h_func=heuristic1)

    # 能够吃到食物
    if path_to_food is not None:
        virtual_game_graph = copy.deepcopy(game_graph)
        move = (path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1])
        for i in range(len(path_to_food) - 1):
            virtual_game_graph.set_aim(path_to_food[i+1][0] - path_to_food[i][0], path_to_food[i+1][1] - path_to_food[i][1])
            virtual_game_graph.move_snake()  # 得到虚拟蛇
        path_to_tail2 = A_star(virtual_game_graph.snake[-1], virtual_game_graph.snake[0],
                               virtual_game_graph, h_func=heuristic1)
        # 虚拟蛇可以到达尾巴，则原蛇可以移动
        if path_to_tail2 is not None:
            return move
        # 虚拟蛇无法到达尾巴，但是原蛇可以到达尾巴
        elif path_to_tail is not None:
            return longest_to_(food, game_graph)

    # 不能吃到食物且原蛇可以到达尾巴
    elif path_to_food is None and path_to_tail is not None:
        return longest_to_(tail, game_graph)

    else:
        right_moves = []
        for move in moves:
            if game_graph.is_valid_move(move):
                right_moves.append(move)
        if len(right_moves) == 0:
            return (0, 1)
        return right_moves[random.randint(0, len(right_moves)-1)]


if __name__ == '__main__':
    game = SnakeGame(pathfinding_func=pathfinding, score_rate=0.9, max_fresh_rate=40)
    game.start_game()