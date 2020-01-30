import os
import pandas as pd
from torch.utils.data import Dataset
from .utils import string_to_datetime


class NicuDataset(Dataset):
    def __init__(self, outcome_csv, root_dir, sampling_strategy='front', transform=None):
        print('outcome_csv', outcome_csv)
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.root_dir = root_dir
        self.transform = transform
        self.sampling_strategy = sampling_strategy

    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        person_id = case["SUBJECT_ID"]
        cohort_start_date = string_to_datetime(case["COHORT_END_DATE"])
        cohort_end_date = string_to_datetime(case["COHORT_START_DATE"])
        label = case["LABEL"]

        measurement_person_csv = os.path.join(self.root_dir, f'clean_{person_id}.csv')
        m_df = pd.read_csv(measurement_person_csv)

        m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] >= cohort_start_date]
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] < cohort_end_date]

        m_df = self._sampling(m_df)
        m_df = self._normalize(m_df)
        m_df = self._fillna(m_df)

        return {"data": m_df, "label": label}

    def _sampling(self, m_df):
        if self.sampling_strategy == "front":
            m_df = m_df.fillna(method='ffill')
        elif self.sampling_strategy == "average":
            front_fill = m_df.fillna(method='ffill')
            back_fill = m_df.fillna(method='bfill')
            m_df = (front_fill + back_fill) / 2
        else:
            raise RuntimeError("Invalid sampling strategy")

        return m_df

    @staticmethod
    def _normalize(m_df):
        return m_df.apply(lambda x: x - x.mean() / x.std(), axis=0)

    @staticmethod
    def _fillna(m_df):
        return m_df.fillna(0)
