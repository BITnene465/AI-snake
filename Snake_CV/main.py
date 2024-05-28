# 导入所需的库
import cv2
from cvzone.HandTrackingModule import HandDetector
from SnakeGame import SnakeGame

def setup_camera(width=1280, height=720):
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 创建视频捕捉对象，参数0表示第一个摄像头
    cap.set(3, width)  # 设置视频宽度
    cap.set(4, height)  # 设置视频高度
    return cap  # 返回配置好的摄像头对象

def main():
    # 主函数
    cap = setup_camera()  # 设置并获取摄像头
    detector = HandDetector(detectionCon=0.8, maxHands=1)  # 初始化手部检测器，检测置信度为0.8，最多检测一只手
    game = SnakeGame("apple.png")  # 初始化贪吃蛇游戏，使用"apple.png"作为苹果的图标

    while True:  # 主循环
        success, img = cap.read()  # 从摄像头读取一帧
        if not success:  # 如果读取失败
            print("Failed to capture image")  # 打印错误信息
            break  # 退出循环

        img = cv2.flip(img, 1)  # 将图像水平翻转，使得显示为镜像
        hands, img = detector.findHands(img, flipType=False)  # 在图像中寻找手部，flipType=False表示不需要再次翻转

        if hands:  # 如果检测到手部
            lmList = hands[0]['lmList']  # 获取第一只手的所有关键点坐标
            pointIndex = lmList[8][0:2]  # 获取食指指尖的x和y1坐标
            img = game.update(img, pointIndex)  # 更新游戏状态，并在图像上绘制游戏界面

        cv2.imshow("Image", img)  # 显示图像
        if cv2.waitKey(1) & 0xFF == 27:  # 每1毫秒检查一次是否按下'ESC'键（ASCII值为27）
            break  # 如果按下，则退出循环

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == "__main__":
    main()
