import argparse

# TODO: Refactoring needed
from src.train import main_train as train
from src.inference import main_inference as inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    parser.add_argument('mode', metavar='RUNNING', type=str, default='train')
    args = parser.parse_args()
    print('Environment: ', args.env)
    print('Mode: ', args.mode)
    if args.mode == 'train':
        train(args.env)
    else:
        pass
        inference(
            args.env,
            ckpt_name='200204_epoch7_attn',
            threshold_strategy="exact",  ## "percentile" or "exact"
            threshold_percentile=100 - 0.59,  ## for threshold_strategy == "percentile"
            threshold_exact=0.5,  ## for threshold_strategy == "exact"
            if_use_log=False,
            logfile='011458.csv'
        )
