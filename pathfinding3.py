# 保留着我的遗憾，dijkstra是不可以的！

import random
from collections import deque
import copy
import heapq


class GameGraph(object):
    def __init__(self, snake, food, edges):
        self.snake = snake
        self.food_x, self.food_y = food
        self.edges = edges
        self.aim_x = 0
        self.aim_y = 0

    def set_food(self, food):
        self.food_x, self.food_y = food

    def set_aim(self, aim_x, aim_y):
        self.aim_x = aim_x
        self.aim_y = aim_y

    def move_snake(self):
        head_move_x = self.snake[-1][0] + self.aim_x
        head_move_y = self.snake[-1][1] + self.aim_y
        self.snake.append((head_move_x, head_move_y))
        if head_move_x != self.food_x or head_move_y != self.food_y:
            self.snake.pop(0)  # 弹出尾部（也即更新蛇蛇）
        print("head:", self.snake[-1], "tail:", self.snake[0], "aim:", self.aim_x, self.aim_y)   # 监测蛇蛇


    def is_inside(self, position):

        if self.edges['xmin'] <= position[0] <= self.edges['xmax'] and self.edges['ymin'] <= position[1] <= self.edges['ymax']:
            return True
        else:
            return False

    def is_valid_move(self, move):
        head_move = (self.snake[-1][0] + move[0], self.snake[-1][1] + move[1])
        if not self.is_inside(head_move) or (head_move in self.snake and head_move != self.snake[0]):
            return False
        return True

    def is_collision(self, position):
        if position == self.snake[0]:   # 刚好是尾部,特判一下
            return False
        if position in self.snake:
            return True
        return False

    def to_tail(self):
        return bfs(self.snake[-1], self.snake[0], self)

    def longest_to_tail(self):
        return dijkstra(self.snake[-1], self.snake[0], self)


def bfs(start, end, game_graph: GameGraph):  # 最重要的函数
    # 使用广度优先搜索查找最短路径
    queue = deque()
    queue.append(start)

    parent = dict()     # 记录每个点的前驱
    parent[start] = None

    while queue:
        current = queue.popleft()

        if current == end:
            break

        for move in [(0, 10), (0, -10), (10, 0), (-10, 0)]:
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


# todo 使用 Dijkstra 算法求最长路
def dijkstra(start: (int, int), end: (int, int), game_graph: GameGraph) -> [(int, int)]:
    # 不在字典里就是没更新，就是负无穷大
    inf = -1000000000
    distances = dict()
    distances[start] = 0

    # 优先队列，需要 heapq 模块进行操作
    priority_que = [(0, start)]

    # flag 数组， python里面就用集合实现吧（
    flag = set()   # 初始时所有点都不在集合内

    # 记录前驱，用于找路径
    parent = dict()
    parent[start] = None

    while priority_que:    # 优先队列非空
        current_distance, current_pos = heapq.heappop(priority_que)   # todo 弹出dis最大，大根堆

        if current_pos in flag:
            continue

        flag.add(current_pos)
        for move in [(0, 10), (0, -10), (10, 0), (-10, 0)]:  # 遍历相邻点
            neighbor = (current_pos[0] + move[0], current_pos[1] + move[1])
            distance = current_distance + 1

            # neighbor 的合法性判断
            if game_graph.is_collision(neighbor) or (not game_graph.is_inside(neighbor)):
                continue

            if distance > distances.get(neighbor, inf):
                distances[neighbor] = distance
                parent[neighbor] = current_pos # 记录前驱
                heapq.heappush(priority_que, (distance, neighbor))

    # 返回长度的版本，但是我们需要路径！！！
    # if end in distances:
    #     return distances[end]
    # return None

    # 返回路径
    if end not in distances:   # 不存在路径
        return None
    path = []
    current = end
    while current != None:
        path.append(current)
        current = parent[current]

    print(path)
    path.reverse()
    return path


def pathfinding(game_graph: GameGraph) -> (int, int):
    moves = [(0, 10), (0, -10), (10, 0), (-10, 0)]
    food = (game_graph.food_x, game_graph.food_y)
    path_to_food = bfs(game_graph.snake[-1], food, game_graph)
    path_to_tail = game_graph.to_tail()
    # 长度为1 特判
    if len(game_graph.snake) == 1:
        return path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1]
    # 能够吃到食物
    if path_to_food is not None:
        virtual_game_graph = copy.deepcopy(game_graph)
        move = (path_to_food[1][0] - path_to_food[0][0], path_to_food[1][1] - path_to_food[0][1])
        virtual_game_graph.aim_x = move[0]
        virtual_game_graph.aim_y = move[1]
        virtual_game_graph.move_snake()  # 得到虚拟蛇
        path_to_tail2 = virtual_game_graph.to_tail()
        # 虚拟蛇可以到达尾巴，则原蛇可以移动
        if path_to_tail2 is not None:
            return move
        # 虚拟蛇无法到达尾巴，但是原蛇可以到达尾巴，向 到尾巴的最长路径前进
        elif path_to_tail is not None:
            longest_path_to_tail = game_graph.longest_to_tail()   # 最短路存在，那么最长路也一定存在
            move = (longest_path_to_tail[1][0] - longest_path_to_tail[0][0],
                    longest_path_to_tail[1][1] - longest_path_to_tail[0][1])
            return move

    # 不能吃到食物且原蛇可以到达尾巴，向远离尾巴的方向前进
    if path_to_food is None and path_to_tail is not None:
        longest_path_to_tail = game_graph.longest_to_tail()
        move = (longest_path_to_tail[1][0] - longest_path_to_tail[0][0],
                longest_path_to_tail[1][1] - longest_path_to_tail[0][1])
        return move

    # Wander 来点随机性
    right_moves = []
    for move in moves:
        if game_graph.is_valid_move(move):
            right_moves.append(move)
    if len(right_moves) == 0:
        return 0, 10      # 这蛇直接撞死吧，没救了
    return right_moves[random.randint(0, len(right_moves)-1)]   # 两端都包括

    # todo 当没法吃到食物并且没法找到尾巴时，采用  算法向远离食物的方向前进


# 测试代码
if __name__ == '__main__':
    game = GameGraph([(0, 0), (0, 10)], (0, 0), {'xmin': -200, 'xmax': 180, 'ymin': -190, 'ymax': 190})
    print(dijkstra((0, 0), (100, 100), game))
