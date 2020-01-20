import argparse

from src.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    args = parser.parse_args()
    print('ENVIRONMENT : ', args.env)
    train(args.env)
