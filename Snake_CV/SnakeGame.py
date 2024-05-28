import math
import random
import cv2
import cvzone
import numpy as np

# 颜色定义
COLOR_SNAKE_HEAD = (30, 160, 75)  # 深绿色
COLOR_SNAKE_BODY = (140, 180, 50)  # 深黄绿色
COLOR_FOOD = (200, 60, 50)   # 深红色
COLOR_BORDER = (205, 133, 63)  # 深黄色 (土黄色)
COLOR_TEXT = (250, 250, 210)  # 亚麻色
COLOR_BACKGROUND = (25, 25, 25)  # 暗灰色



class SnakeGame:
    def __init__(self, pathFood, border_left=50, border_top=20, border_right=1230, border_bottom=700):
        self.points = []  # 蛇的所有点`
        self.lengths = []
        self.curLength = 0
        self.allowedLength = 350
        self.preHead = (0, 0)
        self.score = 0
        self.gameover = False
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.randomFoodLocation()
        # 边界定义
        self.border_left = border_left
        self.border_top = border_top
        self.border_right = border_right
        self.border_bottom = border_bottom

    def randomFoodLocation(self):
        while True:
            new_food_point = random.randint(100, 1000), random.randint(100, 600)

            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            if cv2.pointPolygonTest(pts, new_food_point, False) != 0:
                self.foodPoint = new_food_point
                break

    def putTextWithOutline(self, img, text, position, font_scale, font_thickness, text_color, outline_color):
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, outline_color, font_thickness + 4, lineType=cv2.LINE_AA)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    def handle_game_over(self, imgMain):
        """处理游戏结束的显示（显示为中文）"""
        self.putTextWithOutline(imgMain, "Game Over", (390, 300), 2, 3, COLOR_TEXT, COLOR_BORDER)
        self.putTextWithOutline(imgMain, f'Final Score: {self.score}', (390, 450), 2, 3, COLOR_TEXT, COLOR_BORDER)

    def update_snake_position(self, curHead):
        """更新蛇的位置，并处理吃食物的动画状态"""
        px, py = self.preHead
        cx, cy = curHead
        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.curLength += distance
        self.preHead = cx, cy
        if self.curLength > self.allowedLength:
            for i, length in enumerate(self.lengths):  # 从后往前
                self.curLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.curLength < self.allowedLength:
                    break


    def handle_food_collision(self, curHead):
        """处理蛇吃苹果的逻辑"""
        cx, cy = curHead
        rx, ry = self.foodPoint
        if (rx - self.wFood // 2 < cx < rx + self.wFood // 2 and
                ry - self.wFood // 2 < cy < ry + self.wFood // 2):
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
            print(self.score)

    def draw_elements(self, imgMain):
        """绘制蛇和食物以及边界框，并添加动画效果"""
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    color_gradient = (COLOR_SNAKE_BODY[0] + i % 255, COLOR_SNAKE_BODY[1], COLOR_SNAKE_BODY[2])
                    cv2.line(imgMain, self.points[i - 1], self.points[i], color_gradient, 15)
            cv2.circle(imgMain, self.points[-1], 20, COLOR_SNAKE_HEAD, cv2.FILLED)
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.wFood // 2))
        self.putTextWithOutline(imgMain, f'Score: {self.score}', [80, 80], 1.5, 3, COLOR_TEXT, COLOR_BORDER)
        cv2.rectangle(imgMain, (self.border_left, self.border_top), (self.border_right, self.border_bottom),
                      COLOR_BORDER, 3)

    def check_collision(self, imgMain, curHead):
        """检测蛇与自身的碰撞"""
        cx, cy = curHead
        pts = np.array(self.points[:-2], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
        minDis = cv2.pointPolygonTest(pts, (cx, cy), True)
        if -1.5 < minDis < 1.5:
            self.gameover = True

    def check_border_collision(self, curHead):
        """检测蛇是否触碰到边界"""
        cx, cy = curHead
        if cx < self.border_left or cx > self.border_right or cy < self.border_top or cy > self.border_bottom:
            self.gameover = True

    def update(self, imgMain, curHead):
        """更新游戏状态的主函数"""
        imgMain[:] = COLOR_BACKGROUND  # 设置全局背景颜色
        if self.gameover:
            self.handle_game_over(imgMain)
        else:
            self.update_snake_position(curHead)
            self.handle_food_collision(curHead)
            self.draw_elements(imgMain)
            self.check_collision(imgMain, curHead)
            self.check_border_collision(curHead)
        return imgMain

