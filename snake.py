import copy
import pygame
import random
from typing import Optional, Callable, Tuple
from GameGraph import GameGraph
import pathfinding_greedy   # 只是为了提供一个可测试的函数，可以去掉


class SnakeGame(object):
    def __init__(self, left=1, right=10, top=10, bottom=1, size=40, score_rate=3 / 4, max_fresh_rate=60,
                 pathfinding_func=None, not_display_on_gui=False):
        # 常量区
        self.LEFT, self.RIGHT, self.TOP, self.BOTTOM = left, right, top, bottom
        self.all_food = []  # 所有食物的位置
        self.SIZE = size  # 每个方块的尺寸
        self.LOOSE = score_rate  # 达到 LOOSE*全部方格数 就算赢
        self.MAX_REFRESH_RATE = max_fresh_rate  # 最大刷新率  但是不一定可以达到
        self.NOTDISPLAY = not_display_on_gui

        # pygame GUI 的相关
        self.screen_width, self.screen_height = self.RIGHT - self.LEFT + 5, self.TOP - self.BOTTOM + 6
        self.screen = None

        # 常量 -- 控制游戏区域在 screen 的居中
        self.offsetx = (self.screen_width - (self.RIGHT - self.LEFT)) // 2 - 1
        self.offsety = (self.screen_height - (self.TOP - self.BOTTOM)) // 2 - 1

        # 核心寻路函数
        self.pathfinding: Optional[Callable[[GameGraph], Tuple[int, int]]] = pathfinding_func

        # 核心变量
        self.game_graph = None
        self.snake = []  # 蛇的起始位置
        self.food_x, self.food_y = None, None
        self.aim_x, self.aim_y = None, None  # 蛇的起始方向
        self.score = 0  # 分数


    def start_game(self):
        self.setup_game()

        if not self.NOTDISPLAY:
            self.run_game()
        else:
            while self.move(self.pathfinding(self.game_graph)):
                pass

    def setup_game(self):
        # Pygame 初始化
        if not self.NOTDISPLAY:
            pygame.init()
            self.font = pygame.font.SysFont("Arial", 28, bold=True)
            self.screen = pygame.display.set_mode((self.screen_width * self.SIZE, self.screen_height * self.SIZE))
            pygame.display.set_caption('贪吃蛇')

        # 常量赋值
        self.all_food = []
        for x in range(self.LEFT, self.RIGHT + 1, 1):
            for y in range(self.BOTTOM, self.TOP + 1, 1):
                self.all_food.append((x, y))

        # 核心变量赋值
        self.snake = [(2, 2), (3, 2)]  # list内一定要使用元组
        self.food_x, self.food_y = self.new_food()
        self.aim_x, self.aim_y = 0, 0
        self.score = len(self.snake)  # 初始化分数

        # 此处传入的是蛇蛇的深复制
        self.game_graph = GameGraph(copy.deepcopy(self.snake), (self.food_x, self.food_y),
                                    {'xmin': self.LEFT, 'xmax': self.RIGHT, 'ymin': self.BOTTOM, 'ymax': self.TOP},
                                    self.SIZE)

    def run_game(self):
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                else:
                    if not self.move(self.pathfinding(self.game_graph)):
                        running = False
                        break

            clock.tick(self.MAX_REFRESH_RATE)
        # pygame.quit()

    def draw_square(self, screen, x, y, size, color):
        """绘制小正方形, 代表一格"""
        pygame.draw.rect(screen, color, pygame.Rect((x + self.offsetx) * size, (y + self.offsety) * size, size, size))

    def draw_frame(self, screen):
        """绘制边框"""
        for i in range(self.LEFT - 1, self.RIGHT + 2, 1):
            self.draw_square(screen, i, self.BOTTOM - 1, self.SIZE, pygame.Color('gray'))
            self.draw_square(screen, i, self.TOP + 1, self.SIZE, pygame.Color('gray'))
        for i in range(self.BOTTOM - 1, self.TOP + 2, 1):
            self.draw_square(screen, self.LEFT - 1, i, self.SIZE, pygame.Color('gray'))
            self.draw_square(screen, self.RIGHT + 1, i, self.SIZE, pygame.Color('gray'))

    def change(self, direction):
        """改变蛇的运动方向"""
        x, y = direction
        if (x != -self.aim_x or y != -self.aim_y) and (abs(x) + abs(y) == 1):
            self.aim_x = x
            self.aim_y = y
        else:
            print("朝向改变不合理，请检查你的决策函数")

    def inside(self, head_x, head_y):
        """判断蛇是否在边框内"""
        return self.LEFT <= head_x <= self.RIGHT and self.BOTTOM <= head_y <= self.TOP

    def new_food(self):
        """随机生成食物位置"""
        # 优化食物生成算法
        food = self.all_food.copy()
        for i in self.snake:
            food.remove(i)
        new_food_x, new_food_y = food.pop(random.randint(0, len(food) - 1))
        return new_food_x, new_food_y

    def interpolate_color(self, start_color, end_color, steps):
        """生成两种颜色之间的渐变色"""
        for i in range(steps):
            r = start_color[0] + (end_color[0] - start_color[0]) * i / steps
            g = start_color[1] + (end_color[1] - start_color[1]) * i / steps
            b = start_color[2] + (end_color[2] - start_color[2]) * i / steps
            yield (r, g, b)

    def draw_snake(self, screen):
        """绘制蛇"""
        num_segments = len(self.snake)
        start_color = (255, 128, 128)  # 蛇尾
        end_color = (255, 0, 0)  # 蛇头
        for i, color in enumerate(self.interpolate_color(start_color, end_color, num_segments)):
            self.draw_square(screen, self.snake[i][0], self.snake[i][1], self.SIZE, color)

    def move(self, direction: Tuple[int, int]):
        '''
        每一步更新的地方
        返回值表示游戏是否继续，游戏结束时返回 0,未结束返回 1
        '''
        self.change(direction)  # 做决策时， move函数会等待决策
        # 同步一下 game_graph 的状态
        self.game_graph.set_aim(self.aim_x, self.aim_y)
        self.game_graph.move_snake()

        head_move_x = self.snake[-1][0] + self.aim_x
        head_move_y = self.snake[-1][1] + self.aim_y

        # 判断是否撞到边框或者撞到自己 (判断游戏是否结束)
        if not self.inside(head_move_x, head_move_y) or (
                (head_move_x, head_move_y) in self.snake and (head_move_x, head_move_y) != self.snake[0]):
            
            if not self.NOTDISPLAY:
                self.screen.fill(pygame.Color('white'))
                self.draw_frame(self.screen)
                self.draw_snake(self.screen)
                self.draw_square(self.screen, head_move_x, head_move_y, self.SIZE, pygame.Color('blue'))
                pygame.display.flip()
            print('最终得分: ', self.score)
            return False

        self.snake.append((head_move_x, head_move_y))  # 新的蛇头

        # 判断是否吃到食物以及是否胜利
        if head_move_x == self.food_x and head_move_y == self.food_y:
            self.score += 1  # 每吃到一个食物加1分, 用于显示
            self.game_graph.scoreIncrease()
            print("当前得分：", self.score)
            if len(self.snake) >= len(self.all_food) * self.LOOSE:  # 限制松一点
                print('YOU WIN!')
                print('最终得分: ', self.score)
                return False
            else:
                self.food_x, self.food_y = self.new_food()
                self.game_graph.set_food((self.food_x, self.food_y))
        else:
            self.snake.pop(0)  # 把蛇尾pop掉

        if not self.NOTDISPLAY:
            # 先将 screen 填充为白色背景
            self.screen.fill(pygame.Color('white'))

            # 绘制边框, 蛇和食物
            self.draw_frame(self.screen)
            self.draw_snake(self.screen)
            self.draw_square(self.screen, self.food_x, self.food_y, self.SIZE, pygame.Color('green'))

            # 显示分数
            score_text = self.font.render(f'Score: {self.score}', True, pygame.Color('black'))
            self.screen.blit(score_text, (self.screen_width * self.SIZE // 2 - (1.5)*self.SIZE, self.offsety * self.SIZE // 2 ))

            pygame.display.flip()

            pygame.time.set_timer(pygame.USEREVENT, int(1000 / self.MAX_REFRESH_RATE))  #  1s = 1000ms

        return True



    # 新添加的训练接口, 每次接受一个转向输入 (int, int)
    # 更新游戏状态并且返回一个 (res, gamegraph) , res=True 表示游戏继续; res=False 表示游戏结束
    # gamegraph是更新之后的游戏状态, 是一个 GameGraph 对象

    def move_StepByStep(self, direction: Tuple[int, int]) -> Tuple[bool, GameGraph] :
        """ 一步一步地移动，用于训练模型 """
        res = self.move(direction)
        return res, self.game_graph


    def getGamegraph(self) -> GameGraph:
        """返回的是引用，请不要随意更改值"""
        return self.game_graph


if __name__ == '__main__':
    game = SnakeGame(pathfinding_func=pathfinding_greedy.pathfinding, not_display_on_gui=False)
    game.start_game()
