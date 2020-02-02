import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import string_to_datetime, days_hours_minutes

pd.options.mode.chained_assignment = None  # default='warn'


class CombinedDataset(Dataset):
    def __init__(self, outcome_csv, max_seq_length=256, transform=None):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.dfs = {}
        self.births = {}

    def fill_dfs_and_births(self, dfs, births):
        self.dfs = dfs
        self.births = births

    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        label = 0.0
        if "LABEL" in case:
            label = case["LABEL"]
        person_id = case["SUBJECT_ID"]
        birth_date = self.births[person_id]

        cohort_start_date = string_to_datetime(case["COHORT_START_DATE"])
        start_from_birth = days_hours_minutes(cohort_start_date - string_to_datetime(birth_date))
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])
        end_from_birth = days_hours_minutes(cohort_end_date - string_to_datetime(birth_date))

        c_df = self.dfs[person_id]
        condition = (c_df.index >= start_from_birth) & (c_df.index <= end_from_birth)
        c_df = c_df.loc[condition]
        c_df = np.array(c_df)

        if len(c_df) > self.max_seq_length:
            m_df = c_df[-self.max_seq_length:]
            actual_seq_length = self.max_seq_length
        else:
            actual_seq_length = len(c_df)
            padded_m_df = np.zeros((self.max_seq_length, c_df.shape[1]))
            padded_m_df[:actual_seq_length, :] = c_df
            m_df = padded_m_df

        return torch.tensor(m_df, dtype=torch.float), torch.tensor(actual_seq_length, dtype=torch.long), torch.tensor(
            label, dtype=torch.long)
