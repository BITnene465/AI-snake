"""这是 SnakeGame 的训练接口的使用demo, 直接运行本文件可以看到效果"""
import time
from pathfinding_greedy import pathfinding
from snake import SnakeGame

if __name__ == '__main__':
    game = SnakeGame(not_display_on_gui=False, score_rate=0.5)
    while True:
        game.setup_game()
        gg = game.getGamegraph()
        res = True
        while res:
            res, gg = game.move_StepByStep(direction=pathfinding(gg))
            """  接受此次 gg 的内容，并由此进行训练, 由于gg是引用, 不要操作其值 """
            # time.sleep(0.2)
            print(gg.snake, "score:", gg.score)
        print("game over")

