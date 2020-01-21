import pandas as pd
import torch
from .config import LocalConfig, ProdConfig
from .constants import outcome_cohort_csv


def inference(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    o_df = pd.read_csv(cfg.TEST_DIR + outcome_cohort_csv, encoding='CP949')

    n_dist = torch.distributions.normal.Normal(0.3, 0.1)
    probs = n_dist.sample([len(o_df)]).tolist()

    o_df["LABEL_PROBABILITY"] = probs
    o_df.loc[o_df["LABEL_PROBABILITY"] > 0.5, "LABEL"] = 1
    o_df.loc[o_df["LABEL_PROBABILITY"] <= 0.5, "LABEL"] = 0

    o_df.to_csv(cfg.OUTPUT_DIR + "/output.csv")
