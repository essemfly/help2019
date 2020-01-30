import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import string_to_datetime


class NicuDataset(Dataset):
    def __init__(self, outcome_csv, root_dir, max_seq_length=4096, sampling_strategy='front', transform=None):
        self.o_df = pd.read_csv(outcome_csv, encoding='CP949')
        self.root_dir = root_dir
        self.transform = transform
        self.sampling_strategy = sampling_strategy
        self.max_seq_length = max_seq_length  
        self.person_ids = list(set(o_df.loc[:, "SUBJECT_ID"].values.tolist()))
        self.person_dfs = self.init_person_dfs()
    
    def init_person_dfs(self):
        dfs = {}
        for person_id in self.person_ids:
            measurement_person_csv = os.path.join(self.root_dir, f'clean_{person_id}.csv')
            m_df = pd.read_csv(measurement_person_csv, index_col=0)
            m_df = self._sampling(m_df)
            m_df = self._normalize(m_df)
            m_df = self._fillna(m_df)
            dfs[person_id] = m_df 
        return dfs     
            
    def __len__(self):
        return len(self.o_df)

    def __getitem__(self, idx):
        case = self.o_df.iloc[idx]
        person_id = case["SUBJECT_ID"]
        cohort_start_date = string_to_datetime(case["COHORT_START_DATE"])
        cohort_end_date = string_to_datetime(case["COHORT_END_DATE"])
        label = case["LABEL"]

        #measurement_person_csv = os.path.join(self.root_dir, f'clean_{person_id}.csv')
        #m_df = pd.read_csv(measurement_person_csv, index_col=0)
        m_df = self.person_dfs[person_id]

        m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] >= cohort_start_date]
        m_df = m_df[m_df["MEASUREMENT_DATETIME"] <= cohort_end_date]
        m_df.drop(columns=["MEASUREMENT_DATETIME"], axis=1, inplace=True)
        #m_df = self._sampling(m_df)
        #m_df = self._normalize(m_df)
        #m_df = self._fillna(m_df)
        #m_df = m_df.astype(float)
        m_df = np.array(m_df)
        
        if len(m_df) > self.max_seq_length:
            m_df = m_df[-self.max_seq_length:]  
            actual_seq_length = self.max_seq_length
        else:
            actual_seq_length = len(m_df)
            padded_m_df = np.zeros((self.max_seq_length, m_df.shape[1]))
            padded_m_df[:actual_seq_length, :] = m_df
            m_df = padded_m_df
        
        return torch.tensor(m_df, dtype=torch.float), torch.tensor(actual_seq_length, dtype=torch.long), torch.tensor(label, dtype=torch.long) 
        #m_df = m_df[(len(m_df) - self.max_seq_length):]

        #m_df_list = m_df.values.tolist()
        #for _ in range(self.max_seq_length - len(m_df)):
        #    m_df_list.append(len(m_df.columns) * [0])

        #return torch.tensor(m_df_list), min(len(m_df), self.max_seq_length), label

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
        return m_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    @staticmethod
    def _fillna(m_df):
        return m_df.fillna(0)
