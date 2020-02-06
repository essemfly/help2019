import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv, output_csv, hyperparams, model_config
from .models import NicuModel, ConvLstmLinear
from .datasets.measurement import MeasurementDataset
from .datasets.hourly_sampled import HourlySampledDataset
from .preprocess.sample_by_hour import measure11_dfs, convert_features_to_dataset, measurement_preprocess

def inference(cfg, ckpt_name, threshold_strategy, threshold_percentile, threshold_exact):
    mode = 'test'
    o_df = pd.read_csv(cfg.get_csv_path(outcome_cohort_csv, mode), encoding='CP949')

    batch_size = hyperparams['batch_size']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    sampling_strategy = hyperparams['sampling_strategy']
    max_seq_length = hyperparams['max_seq_len']
    epochs = hyperparams['epochs']
    reverse_pad = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    num_workers = 8 * n_gpu

    #model = NicuModel(device=device)
    model = ConvLstmLinear(device=device)
    model.load_state_dict(torch.load(f'{cfg.VOLUME_DIR}/{ckpt_name}.ckpt'))
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.eval()
    
    prob_preds = []
    
    transforms = None
    testset = HourlySampledDataset(cfg.get_csv_path(outcome_cohort_csv, mode), max_seq_length=max_seq_length,
                                    transform=transforms, reverse_pad=reverse_pad)
    dfs = measure11_dfs(cfg, mode)
    testset.fill_dfs(dfs)
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    print(hyperparams)
    print(model_config)
    print(model)
    
    for x, x_len, _ in tqdm(testloader, desc="Evaluating"):
        x = x.to(device)
        x_len = x_len.to(device)
        actual_batch_size = x.size()
        with torch.no_grad():
            if reverse_pad:
                outputs = model(x)
            else:
                if actual_batch_size[0] == batch_size:
                    outputs = model(x, x_len)
                else:
                    x_padding = torch.zeros(
                        (batch_size - actual_batch_size[0], actual_batch_size[1], actual_batch_size[2])).to(device)
                    x_len_padding = torch.ones(batch_size - actual_batch_size[0], dtype=torch.long).to(device)
                    outputs = model(torch.cat((x, x_padding)), torch.cat((x_len, x_len_padding)))
                    outputs = outputs[:actual_batch_size[0]]
            prob = torch.sigmoid(outputs)
        if len(prob_preds) == 0:
            prob_preds.append(prob.detach().cpu().numpy())
        else:
            prob_preds[0] = np.append(prob_preds[0], prob.detach().cpu().numpy(), axis=0)
    prob_preds = prob_preds[0]

    make_output(cfg, o_df, prob_preds, threshold_strategy, threshold_percentile, threshold_exact, if_savetolog=True,
                if_summary=True)


def inference_with_threshold(cfg, logfile, threshold_strategy, threshold_percentile, threshold_exact):
    o_df = pd.read_csv(cfg.LOG_DIR + '/' + logfile, encoding='CP949')
    prob_preds = o_df["LABEL_PROBABILITY"]

    make_output(cfg, o_df, prob_preds, threshold_strategy, threshold_percentile, threshold_exact, if_savetolog=False,
                if_summary=True)


def make_output(cfg, o_df, prob_preds, threshold_strategy, threshold_percentile, threshold_exact, if_savetolog,
                if_summary):
    label_preds = np.zeros_like(prob_preds)
    threshold = np.percentile(prob_preds, threshold_percentile, interpolation="nearest") if (
            threshold_strategy == "percentile") else threshold_exact

    o_df["LABEL_PROBABILITY"] = prob_preds
    o_df["LABEL"] = label_preds.astype(int)
    o_df.loc[o_df["LABEL_PROBABILITY"] > threshold, "LABEL"] = 1

    if (if_savetolog):
        save_to_log(cfg, o_df)
    if (if_summary):
        inference_summary(o_df, threshold_strategy, threshold, threshold_percentile)

    o_df.to_csv(cfg.OUTPUT_DIR + output_csv,
                columns=["LABEL", "LABEL_PROBABILITY"],
                index=False)


def save_to_log(cfg, o_df):
    logfile = cfg.LOG_DIR + '/' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    o_df.to_csv(logfile)
    print("Saved probability for further threshold tailoring as : ", logfile)


def inference_summary(o_df, threshold_strategy, threshold, threshold_percentile):
    print("### INFERENCE SUMMARY ###")
    print("Total lengths : ", o_df["LABEL"].shape[0], o_df["LABEL_PROBABILITY"].shape[0])
    print("Estimated threshold : ", threshold)
    if (threshold_strategy == "percentile"):
        print("  - which was set by percentile : ", threshold_percentile)
    else:
        print("  - which was set by : EXACT VALUE")
    print("Number of inferred positives : ", o_df["LABEL"].sum())
    print("Mean of probability : ", o_df["LABEL_PROBABILITY"].mean())
    print("Median of probability : ", o_df["LABEL_PROBABILITY"].median())
    print("Min of probability : ", o_df["LABEL_PROBABILITY"].min())
    print("Max of probability : ", o_df["LABEL_PROBABILITY"].max())


def main_inference(env, ckpt_name, threshold_strategy, threshold_percentile, threshold_exact, if_use_log, logfile):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Num threads : ", torch.get_num_threads())
    if (if_use_log):
        print("Inference using previous probability log : ", logfile)
        inference_with_threshold(cfg, logfile, threshold_strategy, threshold_percentile, threshold_exact)
    else:
        print("Inference function runs with : ", ckpt_name)
        inference(cfg, ckpt_name, threshold_strategy, threshold_percentile, threshold_exact)
