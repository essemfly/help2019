import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import string_to_datetime

pd.options.mode.chained_assignment = None  # default='warn'


class HourlySampledDataset(Dataset):
    def __init__(self, outcome_csv, max_seq_length=12, transform=None, reverse_pad=True):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.dfs = {}
        self.births = {}
        self.reverse_pad = reverse_pad

    def fill_dfs(self, dfs):
        self.dfs = dfs

    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        label = 0.0
        if "LABEL" in case:
            label = case["LABEL"]
        person_id = case["SUBJECT_ID"]

        cohort_start_date = string_to_datetime(case["COHORT_START_DATE"])
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])

        m_df = self.dfs[person_id]
        condition = (m_df["MEASUREMENT_DATETIME"] > cohort_start_date) & (
                m_df["MEASUREMENT_DATETIME"] <= cohort_end_date)
        m_df = m_df[condition]
        m_df.drop(columns=["MEASUREMENT_DATETIME"], axis=1, inplace=True)
        m_df = np.array(m_df)

        if len(m_df) > self.max_seq_length:
            m_df = m_df[-self.max_seq_length:]
            actual_seq_length = self.max_seq_length
        else:
            actual_seq_length = len(m_df)
            padded_m_df = np.zeros((self.max_seq_length, m_df.shape[1]))
            if self.reverse_pad:
                padded_m_df[-actual_seq_length:, :] = m_df
            else:
                padded_m_df[:actual_seq_length, :] = m_df
            m_df = padded_m_df

        return torch.tensor(m_df, dtype=torch.float), torch.tensor(actual_seq_length, dtype=torch.long), torch.tensor(
            label, dtype=torch.long)
