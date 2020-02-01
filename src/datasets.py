import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import string_to_datetime, days_hours_minutes

pd.options.mode.chained_assignment = None  # default='warn'

class NicuDataset(Dataset):
    def __init__(self, outcome_csv, max_seq_length=4096, transform=None):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.person_dfs = {}
        self.births = {}

    def fill_people_dfs_and_births(self, dfs, births):
        self.person_dfs = dfs
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
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])

        start_from_birth = days_hours_minutes(cohort_start_date - string_to_datetime(birth_date))
        end_from_birth = days_hours_minutes(cohort_end_date - string_to_datetime(birth_date))

        m_df = self.person_dfs[person_id]
        condition = (m_df["TIME_FROM_BIRTH"] >= start_from_birth) & (m_df["TIME_FROM_BIRTH"] <= end_from_birth)
        m_df = m_df[condition]
        m_df.drop(columns=["TIME_FROM_BIRTH"], axis=1, inplace=True)

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
