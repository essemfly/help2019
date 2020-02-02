import pandas as pd
from .constants import MEASUREMENT_SOURCE_VALUE_USES, preprocess_csv, person_csv
from .utils import get_person_ids


def exupperlowers(measurement_df):
    ref_m = pd.read_csv('sample' + preprocess_csv, encoding='CP949')
    df = measurement_df
    for i in range(len(MEASUREMENT_SOURCE_VALUE_USES)):
        if (ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['METHODS'] == 1):
            lower = ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['LOWER']
            upper = ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['UPPER']
            df = df.drop(df[(df['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]) & (
                    df['VALUE_SOURCE_VALUE'] <= lower)].index)
            df = df.drop(df[(df['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]) & (
                    df['VALUE_SOURCE_VALUE'] > upper)].index)
    return df.reset_index()


def preprocess(cfg, mode, sampling_strategy):
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode))
    person_ids = get_person_ids(p_df)
    for person_id in person_ids:
        print('person_id: ', person_id)
        m_df = pd.read_csv(cfg.get_divided_file_path(mode, person_id), index_col=0)
        from_birth_df = m_df["TIME_FROM_BIRTH"]
        m_df.drop(columns=["MEASUREMENT_DATETIME", "TIME_FROM_BIRTH"], axis=1, inplace=True)
        m_df = _sampling(m_df, sampling_strategy)
        m_df = _normalize(m_df)
        m_df = _fillna(m_df)
        m_df["TIME_FROM_BIRTH"] = from_birth_df
        m_df.to_pickle(cfg.get_sampled_file_path(mode, sampling_strategy, person_id))


def _sampling(m_df, sampling_strategy):
    if sampling_strategy == "front":
        m_df = m_df.fillna(method='ffill')
    elif sampling_strategy == "average":
        front_fill = m_df.fillna(method='ffill')
        back_fill = m_df.fillna(method='bfill')
        m_df = (front_fill + back_fill) / 2
    else:
        raise RuntimeError("Invalid sampling strategy")

    return m_df


def _normalize(m_df):
    return m_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def _fillna(m_df):
    return m_df.fillna(0)
