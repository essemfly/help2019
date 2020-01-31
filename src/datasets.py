import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import string_to_datetime, get_person_ids


class NicuDataset(Dataset):
    def __init__(self, outcome_csv, person_csv, root_dir, max_seq_length=4096, sampling_strategy='front',
                 transform=None):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.person_dfs = self._load_person_dfs(person_csv, sampling_strategy)

    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        person_id = case["SUBJECT_ID"]
        cohort_start_date = string_to_datetime(case["COHORT_START_DATE"])
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])
        label = case["LABEL"]

        m_df = self.person_dfs[person_id]

        m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] >= cohort_start_date]
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] <= cohort_end_date]
        m_df.drop(columns=["MEASUREMENT_DATETIME"], axis=1, inplace=True)

        m_df = np.array(m_df)

        if len(m_df) > self.max_seq_length:
            m_df = m_df[-self.max_seq_length:]
            actual_seq_length = self.max_seq_length
        else:
            actual_seq_length = len(m_df)
            padded_m_df = np.zeros((self.max_seq_length, m_df.shape[1]))
            padded_m_df[:actual_seq_length, :] = m_df
            m_df = padded_m_df

        return torch.tensor(m_df, dtype=torch.float), torch.tensor(actual_seq_length, dtype=torch.long), torch.tensor(
            label, dtype=torch.long)

    def _load_person_dfs(self, person_csv, sampling_strategy):
        p_df = pd.read_csv(person_csv, encoding='CP949')
        person_ids = get_person_ids(p_df)
        dfs = {}
        for person_id in person_ids:
            dfs[person_id] = pd.read_pickle(os.path.join(self.root_dir, f'clean_{sampling_strategy}_{person_id}.pkl'))
        return dfs
