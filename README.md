# 贪吃蛇游戏和路径规划

这个项目包括两部分代码：

- 一个简单的贪吃蛇游戏，即 `snake.py` 文件
- 一个用于路径规划的AI，即 `pathfinding.py` 模块

> 贪吃蛇游戏是一个经典的游戏，而路径规划AI使贪吃蛇更加智能

## 贪吃蛇游戏

### 游戏简介

- 本游戏基于Python开发，使用了Turtle图形库来实现游戏界面。
- 目标是通过控制蛇的移动，吃食物，尽量使蛇变得更长，直到达到游戏胜利条件。

### 游戏规则

- 初始时，蛇位于游戏界面的中央，而食物随机生成在游戏界面的空白位置。
- 玩家（程序）可以控制蛇的方向，使其移动。
- 游戏中，蛇的头部碰到游戏边界或者撞到自己的身体会导致游戏结束。
- 当蛇吃到食物时，它的身体会变长，同时新的食物会随机生成。
- 场上只会出现一个食物

### 使用说明

- 运行主程序的Python脚本即可启动游戏。
- 控制蛇的方向使用键盘上的箭头键。
- 游戏界面上，蛇头用红色表示，蛇的身体用黑色表示，食物用绿色表示。

## 路径规划AI

### 项目简介

- 路径规划AI的主要功能是协助贪吃蛇游戏中的蛇智能地移动，以吃到食物或避免碰撞。

### 主要功能

- `GameGraph` 类表示游戏图形，包括蛇、食物、和游戏界限。
- 该AI使用广度优先搜索（BFS）来查找最短路径，以确定蛇的下一个移动方向。
- 它还考虑了多种情况，例如蛇能否吃到食物、是否能到达蛇的尾巴、以及当没有明确路径时采用的漫游策略。
- 由于水平有限，当前版本的 AI 仍然会出现死循环的现象，算法仍需改进

## 如何使用

- 若要使用路径规划AI，可以创建一个 `GameGraph` 实例并调用 `pathfinding` 函数，以获取下一个最佳移动方向。
- 该AI可以自动决定蛇的下一步，观赏即可。

## 注意事项

- 本项目的代码基于Python编写，使用了Turtle图形库，因此在运行之前，请确保您已安装所需的库。

感谢您阅读本ReadMe文件，希望这个贪吃蛇游戏和路径规划AI对您有所帮助，同时也能为您提供娱乐和学习的乐趣。如有任何疑问或建议，请随时联系我们。祝您玩得开心！
