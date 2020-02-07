import argparse

# TODO: Refactoring needed
from src.train import main_train as train
from src.inference import main_inference as inference
from src.constants import hyperparams, model_config

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
        if hyperparams['mixout_epochs'] == 0:
            ckpt_name = f'{model_config["model_name"]}_epoch{hyperparams["epochs"]}_{hyperparams["finetuning_epochs"]}'
        else:
            ckpt_name = f'{model_config["model_name"]}_epoch{hyperparams["epochs"]}_{hyperparams["finetuning_epochs"]}_{hyperparams["mixout_epochs"]}'
        inference(
            args.env,
            ckpt_name=ckpt_name,
            threshold_strategy="exact",  ## "percentile" or "exact"
            threshold_percentile=100 - 0.59,  ## for threshold_strategy == "percentile"
            threshold_exact=0.5,  ## for threshold_strategy == "exact"
            if_use_log=False,
            logfile='011458.csv'
        )
