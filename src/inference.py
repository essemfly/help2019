import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, outcome_cohort_csv
from .models import LSTM
from .datasets import NicuDataset


def inference(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig

    o_df = pd.read_csv(cfg.TEST_DIR + outcome_cohort_csv, encoding='CP949')
    batch_size = 1
    input_size = len(MEASUREMENT_SOURCE_VALUE_USES) + 1
    hidden_size = 128
    sampling_strategy = 'front'
    num_workers = 4
    num_labels = 2
    model_name = 'epoch5'

    model = LSTM(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size,
                 num_labels=num_labels)

    model.load_state_dict(torch.load(f'{cfg.VOLUME_DIR}/{model_name}.ckpt'))
    model.eval()

    label_preds = []

    transforms = None
    testset = NicuDataset(cfg.TEST_DIR + outcome_cohort_csv, cfg.VOLUME_DIR, sampling_strategy=sampling_strategy,
                          transform=transforms)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for x, x_len, _ in testloader:
        with torch.no_grad():
            logits = torch.nn.functional.softmax(model(x, x_len))
        if len(label_preds) == 0:
            label_preds.append(logits.detach().cpu().numpy())
        else:
            label_preds[0] = np.append(label_preds[0], logits.detach().cpu().numpy(), axis=0)
    label_preds = label_preds[0]
    prob_preds = label_preds[:, 1]
    label_preds = np.argmax(label_preds, axis=1)

    o_df["LABEL_PROBABILITY"] = prob_preds
    o_df["LABEL"] = label_preds
    o_df.to_csv(cfg.OUTPUT_DIR + "/output.csv")
