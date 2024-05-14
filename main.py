import argparse
import snake

def main():
    parser = argparse.ArgumentParser(description='Snake game with command line arguments')
    parser.add_argument('-l', '--left', type=int, default=-4, help='left boundary')
    parser.add_argument('-r', '--right', type=int, default=4, help='right boundary')
    parser.add_argument('-t', '--top', type=int, default=4, help='top boundary')
    parser.add_argument('-b', '--bottom', type=int, default=-4, help='bottom boundary')
    parser.add_argument("--size", type=int, default=40, help='length of square')
    parser.add_argument("--rate", type=int, default=60, help='fresh rate')
    parser.add_argument("--score", type=float, default=11/12, help="rating of score when winning")

    args = parser.parse_args()
    snake.start(args.left, args.right, args.top, args.bottom, args.size, args.score, args.rate)

if __name__ == '__main__':
    main()

