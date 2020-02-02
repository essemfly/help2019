import argparse

# TODO: Refactoring needed
from src.train import main_train as train
from src.train_condition import main_train as train_condition
from src.train_combined import main_train as train_combined
from src.inference import main_inference
from src.inference_condition import main_inference as inference_condition
from src.inference_combined import main_inference as inference_combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    parser.add_argument('mode', metavar='RUNNING', type=str, default='train')
    args = parser.parse_args()
    print('Environment: ', args.env)
    print('Mode: ', args.mode)
    if args.mode == 'train':
        train_combined(args.env)
    else:
        inference_combined(args.env,
                           ckpt_name='200203_combined_epoch30_base',
                           threshold_strategy="exact",  ## "percentile" or "exact"
                           threshold_percentile=100 - 0.59,  ## for threshold_strategy == "percentile"
                           threshold_exact=0.5,  ## for threshold_strategy == "exact"
                           if_use_log=False,
                           logfile='011458.csv')
