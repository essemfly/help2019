import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import string_to_datetime, get_person_ids, days_hours_minutes


class NicuDataset(Dataset):
    def __init__(self, outcome_csv, person_csv, root_dir, max_seq_length=4096, sampling_strategy='front',
                 transform=None):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.person_dfs, self.births = self._load_person_dfs(person_csv, sampling_strategy)

    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        person_id = case["SUBJECT_ID"]
        birth_date = self.births[person_id]

        cohort_start_date = string_to_datetime(case["COHORT_START_DATE"])
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])

        start_from_birth = days_hours_minutes(cohort_start_date - string_to_datetime(birth_date))
        end_from_birth = days_hours_minutes(cohort_end_date - string_to_datetime(birth_date))

        label = case["LABEL"]

        m_df = self.person_dfs[person_id]
        m_df.drop(columns=["MEASUREMENT_DATETIME"], axis=1, inplace=True)
        m_df = m_df[m_df["TIME_FROM_BIRTH"] >= start_from_birth]
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] <= end_from_birth]

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
        return dfs, p_df[["PERSON_ID", "BIRTH_DATETIME"]].set_index("PERSON_ID").to_dict(orient='dict')[
            "BIRTH_DATETIME"]
