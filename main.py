import argparse

from src.train import main
from src.inference import inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    parser.add_argument('mode', metavar='RUNNING', type=str, default='train')
    args = parser.parse_args()
    print('Environment: ', args.env)
    print('Mode: ', args.mode)
    if args.mode == 'train':
        main(args.env)
    else:
        inference(args.env)