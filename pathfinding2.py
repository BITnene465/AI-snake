from collections import deque


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
        # print("head:", self.snake[-1], "tail:", self.snake[0], "aim:", self.aim_x, self.aim_y)   # 监测蛇蛇


    def is_inside(self, position):

        if self.edges['xmin'] <= position[0] <= self.edges['xmax'] and self.edges['ymin'] <= position[1] <= self.edges['ymax']:
            return True
        else:
            return False

    def is_collision(self, position):
        if position == self.snake[0]:   # 刚好是尾部,特判一下
            return False
        if position in self.snake:
            return True
        return False


def pathfinding(game_graph: GameGraph) -> (int, int):
    def is_valid_move(move: (int, int), current: (int, int), game_graph: GameGraph):
        x, y = current[0] + move[0], current[1] + move[1]
        next_position = (x, y)
        return game_graph.is_inside(next_position) and not game_graph.is_collision(next_position)

    def find_path(start: (int, int), goal: (int, int), game_graph: GameGraph):
        # 使用广度优先搜索查找最短路径
        queue = deque()
        queue.append(start)

        parent = {}
        parent[start] = None

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for move in [(0, 10), (0, -10), (10, 0), (-10, 0)]:
                if is_valid_move(move, current, game_graph):
                    x, y = current[0] + move[0], current[1] + move[1]
                    next_position = (x, y)
                    if next_position not in parent:
                        queue.append(next_position)
                        parent[next_position] = current

        if goal not in parent:
            return None

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path

    def can_reach_tail(start: (int, int), game_graph: GameGraph):
        # 检查是否能够找到蛇尾的路径
        path = find_path(start, game_graph.snake[0], game_graph)
        return path is not None

    def is_farthest_from_food(start: (int, int), game_graph: GameGraph):
        # 寻找离食物最远的位置
        farthest_position = start
        max_distance = 0
        for x in range(game_graph.edges['xmin'], game_graph.edges['xmax'] + 1, 10):
            for y in range(game_graph.edges['ymin'], game_graph.edges['ymax'] + 1, 10):
                if not game_graph.is_collision((x, y)):
                    path = find_path(start, (x, y), game_graph)
                    if path:
                        distance = len(path)
                        if distance > max_distance:
                            max_distance = distance
                            farthest_position = (x, y)
        return farthest_position

    start = game_graph.snake[-1]
    goal = (game_graph.food_x, game_graph.food_y)

    # 如果可以吃到食物且找到尾巴，按规则最短路径吃食物
    if can_reach_tail(start, game_graph) and can_reach_tail(goal, game_graph):
        path = find_path(start, goal, game_graph)
        if path:
            next_move = (path[1][0] - start[0], path[1][1] - start[1])
            return next_move

    # 如果无法按规则最短路径吃食物，按不规则最短路径吃食物
    if can_reach_tail(start, game_graph):
        farthest_position = is_farthest_from_food(goal, game_graph)
        path = find_path(start, farthest_position, game_graph)
        if path:
            next_move = (path[1][0] - start[0], path[1][1] - start[1])
            return next_move

    # 如果可以到达自己的尾巴并且移动一步不会导致无法到达尾巴，选择离食物最远的位置移动
    tail_position = game_graph.snake[0]
    if can_reach_tail(start, game_graph) and can_reach_tail((start[0] + 10, start[1]), game_graph):
        farthest_position = is_farthest_from_food(tail_position, game_graph)
        path = find_path(start, farthest_position, game_graph)
        if path:
            next_move = (path[1][0] - start[0], path[1][1] - start[1])
            return next_move

    # 如果以上情况都不满足，就听天由命
    current_head_x, current_head_y = game_graph.snake[-1]
    moves = [(0, 10), (0, -10), (10, 0), (-10, 0)]
    for move in moves:
        if is_valid_move(move, (current_head_x, current_head_y), game_graph):
            return move
