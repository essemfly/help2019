import pandas as pd
import numpy as np
import torch
from ..constants import CONDITION_SOURCE_VALUE_USES, MEASUREMENT_SOURCE_VALUE_USES
from torch.utils.data import Dataset
from src.utils import string_to_datetime, days_hours_minutes

pd.options.mode.chained_assignment = None  # default='warn'

class AttentionDataset(Dataset):
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
        target = (c_df.index >= start_from_birth) & (c_df.index <= end_from_birth)
        c_df = c_df.loc[target]
        time = c_df.index.values.reshape(-1, 1)
        condition = np.array(c_df[CONDITION_SOURCE_VALUE_USES])
        measurement = np.array(c_df[MEASUREMENT_SOURCE_VALUE_USES])
        
        if len(c_df) > self.max_seq_length:
            measurement = measurement[-self.max_seq_length:]
            condition = condition[-self.max_seq_length:]
            time = time[-self.max_seq_length:]
            actual_seq_length = self.max_seq_length
        else:
            actual_seq_length = len(c_df)
            padded_measurement = np.zeros((self.max_seq_length, measurement.shape[1]))
            padded_condition = np.zeros((self.max_seq_length, condition.shape[1]))
            padded_time = np.zeros((self.max_seq_length, 1))
            padded_measurement[:actual_seq_length, :] = measurement
            padded_condition[:actual_seq_length, :] = condition
            padded_time[:actual_seq_length, :] = time
            
            measurement = padded_measurement
            condition = padded_condition
            time = padded_time
            
        return torch.tensor(time, dtype=torch.float), torch.tensor(measurement, dtype=torch.float), torch.tensor(condition, dtype=torch.float), torch.tensor(label, dtype=torch.long)
