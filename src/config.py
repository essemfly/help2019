import pandas as pd
from .utils import get_person_ids
from .constants import person_csv


class BaseConfig:
    env = 'localhost'
    TRAIN_DIR = 'sample'
    VOLUME_DIR = 'output'
    LOG_DIR = 'logs'
    TEST_DIR = 'sample'
    OUTPUT_DIR = 'output'

    @classmethod
    def get_csv_path(cls, csv, mode):
        if mode == 'train':
            return cls.TRAIN_DIR + csv
        elif mode == 'test':
            return cls.TEST_DIR + csv
        else:
            raise RuntimeError("Invalid mode for train or test")

    @classmethod
    def get_divided_file_path(cls, mode, person_id):
        if mode != 'train' and mode != 'test':
            raise RuntimeError("Invalid mode for train or test")
        return f'{cls.VOLUME_DIR}/clean_{str(person_id)}_{mode}.csv'

    @classmethod
    def get_sampled_file_path(cls, mode, sampling_strategy, person_id):
        if mode != 'train' and mode != 'test':
            raise RuntimeError("Invalid mode for train or test")
        return f'{cls.VOLUME_DIR}/clean_{sampling_strategy}_{str(person_id)}_{mode}.pkl'

    @classmethod
    def get_condition_file_path(cls, mode, person_id):
        if mode != 'train' and mode != 'test':
            raise RuntimeError("Invalid mode for train or test")
        return f'{cls.VOLUME_DIR}/condition_{str(person_id)}_{mode}.pkl'

    @classmethod
    def load_person_dfs_births(cls, mode, sampling_strategy):
        p_df = pd.read_csv(cls.get_csv_path(person_csv, mode), encoding='CP949')
        person_ids = get_person_ids(p_df)
        dfs = {}
        for person_id in person_ids:
            dfs[person_id] = pd.read_pickle(cls.get_sampled_file_path(mode, sampling_strategy, person_id))
        return dfs, p_df[["PERSON_ID", "BIRTH_DATETIME"]].set_index("PERSON_ID").to_dict(orient='dict')[
            "BIRTH_DATETIME"]

    @classmethod
    def load_condition_dfs_births(cls, mode):
        p_df = pd.read_csv(cls.get_csv_path(person_csv, mode), encoding='CP949')
        person_ids = get_person_ids(p_df)
        dfs = {}
        for person_id in person_ids:
            dfs[person_id] = pd.read_pickle(cls.get_condition_file_path(mode, person_id))
            return dfs, p_df[["PERSON_ID", "BIRTH_DATETIME"]].set_index("PERSON_ID").to_dict(orient='dict')[
                "BIRTH_DATETIME"]


class LocalConfig(BaseConfig):
    env = 'localhost'
    TRAIN_DIR = 'sample'
    VOLUME_DIR = 'output'
    LOG_DIR = 'logs'
    TEST_DIR = 'sample'
    OUTPUT_DIR = 'output'


class ProdConfig(BaseConfig):
    env = 'localhost'
    TRAIN_DIR = '/data/train'
    VOLUME_DIR = '/data/volume'
    LOG_DIR = '/data/volume/logs'
    TEST_DIR = '/data/test'
    OUTPUT_DIR = '/data/output'
