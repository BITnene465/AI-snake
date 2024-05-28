class GameGraph(object):
    DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    VISION_DIRS = [[0, -1], [1, -1], [1, 0], [1, 1],
                   [0, 1], [-1, 1], [-1, 0], [-1, -1]]

    def __init__(self, snake: [(int, int)], food: (int, int), edges: {}, square_size: int) -> None:
        self.snake = snake
        self.food_x, self.food_y = food
        self.edges = edges
        self.size = square_size
        self.aim_x = 0
        self.aim_y = 0
        self.edges = edges
        self.graph_size = (edges['xmax'] - edges['xmin'] + 1)*(edges['ymax'] - edges['ymin'] + 1)
        self.score = len(snake)
        self.head_index = 0

    def set_food(self, food: (int, int)):
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

        if (self.edges['xmin'] <= position[0] <= self.edges['xmax'] and
                self.edges['ymin'] <= position[1] <= self.edges['ymax']):
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

    def GetFood(self):
        return self.food_x, self.food_y

    def GetAim(self):
        return self.aim_x, self.aim_y

    def GetEdge(self):
        return self.edges

    def scoreIncrease(self):
        self.score += 1

    def GetScore(self):
        return self.score

    def GetSnake(self):
        return self.snake

    def to_input_vector(self):
        """输出为一个长度为12的向量, 采用绝对方向"""
        head = self.snake[-1]
        food_x, food_y = self.food_x, self.food_y

        # 蛇头方向（独热编码）
        direction = (head[0] - self.snake[-2][0], head[1] - self.snake[-2][1])
        head_dir_idx = GameGraph.DIRECTIONS.index(direction)
        head_dir = [0] * 4
        head_dir[head_dir_idx] = 1

        # 食物方向的编码（使用独热码，类似 get_state）
        food_direction = [0] * 4
        if food_y < head[1]:
            food_direction[0] = 1  # 食物在上
        elif food_y > head[1]:
            food_direction[2] = 1  # 食物在下
        if food_x > head[0]:
            food_direction[1] = 1  # 食物在右
        elif food_x < head[0]:
            food_direction[3] = 1  # 食物在左

        # 周围的障碍物感知（同 get_state）
        obstacles = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 左, 右, 上, 下
            nx, ny = head[0] + dx, head[1] + dy
            if 0 <= nx < self.edges['xmax'] and 0 <= ny < self.edges['ymax']:
                if (nx, ny) in self.snake:
                    obstacles.append(1)  # 见到自己的身体
                else:
                    obstacles.append(0)
            else:
                obstacles.append(1)  # 见到墙壁

        # 组合所有特征
        state = head_dir + food_direction + obstacles
        return state

    def to_input_vector2(self):
        """输出为一个长度为  3x8 = 24 的向量, 采用相对方向"""
        pass
