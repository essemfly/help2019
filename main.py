import argparse

from src.train import main_train as train
from src.train_condition import main_train as train_condition
from src.inference import main_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    parser.add_argument('mode', metavar='RUNNING', type=str, default='train')
    args = parser.parse_args()
    print('Environment: ', args.env)
    print('Mode: ', args.mode)
    if args.mode == 'train':
        train_condition(args.env)
    else:
        main_inference(args.env,
                       ckpt_name='200203_condition_epoch100_base',
                       threshold_strategy="exact",  ## "percentile" or "exact"
                       threshold_percentile=100 - 0.59,  ## for threshold_strategy == "percentile"
                       threshold_exact=0.5,  ## for threshold_strategy == "exact"
                       if_use_log=False,
                       logfile='011458.csv')
