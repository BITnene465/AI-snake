import snake
import sys

if __name__ == '__main__':
    args = sys.argv
    count = len(args)
    if count == 5:
        snake.start(eval(args[1]), eval(args[2]), eval(args[3]), eval(args[4]))
    elif count == 1:
        snake.start()
    else:
        print("请输入 4个或 0个参数")
