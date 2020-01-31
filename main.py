import argparse

from src.train import main_train
from src.inference import main_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    parser.add_argument('mode', metavar='RUNNING', type=str, default='train')
    args = parser.parse_args()
    print('Environment: ', args.env)
    print('Mode: ', args.mode)
    if args.mode == 'train':
        main_train(args.env)
    else:
        main_inference(args.env, '200131_epoch5_front')
