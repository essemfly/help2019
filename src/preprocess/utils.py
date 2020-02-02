import pandas as pd
from src.constants import MEASUREMENT_SOURCE_VALUE_USES, preprocess_csv


def _exupperlowers(measurement_df):
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
