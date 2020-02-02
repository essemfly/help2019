import pandas as pd
from copy import deepcopy
from src.utils import get_person_ids, get_birth_dates, days_hours_minutes, string_to_datetime
from src.constants import MEASUREMENT_SOURCE_VALUE_USES, CONDITION_SOURCE_VALUE_USES, measurement_csv, person_csv, \
    condition_csv
from src.preprocess.utils import _exupperlowers, _sampling, _normalize, _fillna


def measure_divide(m_df, person_id, birth_date, sampling_strategy):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    m_df = _exupperlowers(m_df)
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)

    records = []
    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"RECORD_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        elif new_personal_record["RECORD_DATETIME"] != row["MEASUREMENT_DATETIME"]:
            records.append(new_personal_record)
            new_personal_record = {"RECORD_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        if row["MEASUREMENT_SOURCE_VALUE"] in MEASUREMENT_SOURCE_VALUE_USES:
            new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        records.append(new_personal_record)

    df = pd.DataFrame(records)
    columns = list(df.columns)
    for source in MEASUREMENT_SOURCE_VALUE_USES:
        if source not in columns:
            df[source] = None

    from_birth_df = df["TIME_FROM_BIRTH"]
    df.drop(columns=["TIME_FROM_BIRTH", "RECORD_DATETIME"], axis=1, inplace=True)
    df = _sampling(df, sampling_strategy)
    df = _normalize(df)
    df = _fillna(df)
    df["TIME_FROM_BIRTH"] = from_birth_df

    return df


def measure_divide_from_pkl(cfg, mode, sampling_strategy, person_id):
    df = pd.read_pickle(cfg.get_sampled_file_path(mode, sampling_strategy, person_id))
    df = df[["TIME_FROM_BIRTH"] + MEASUREMENT_SOURCE_VALUE_USES]
    return df


def condition_divide(c_df, person_id, birth_date):
    c_df = c_df[c_df["PERSON_ID"] == person_id]
    c_df.loc[:, "CONDITION_START_DATETIME"] = pd.to_datetime(c_df["CONDITION_START_DATETIME"],
                                                             format="%Y-%m-%d %H:%M")
    c_df.sort_values("CONDITION_START_DATETIME", inplace=True)

    records = []
    new_personal_record = {condition: 0 for condition in CONDITION_SOURCE_VALUE_USES}
    new_personal_record["RECORD_DATETIME"] = birth_date
    new_personal_record["TIME_FROM_BIRTH"] = 0

    for idx, row in c_df.iterrows():
        if new_personal_record["RECORD_DATETIME"] != row["CONDITION_START_DATETIME"]:
            if row["CONDITION_SOURCE_VALUE"] in new_personal_record:
                records.append(deepcopy(new_personal_record))
                new_personal_record["RECORD_DATETIME"] = row["CONDITION_START_DATETIME"]
                new_personal_record["TIME_FROM_BIRTH"] = days_hours_minutes(
                    row["CONDITION_START_DATETIME"] - birth_date)

        if row["CONDITION_SOURCE_VALUE"] in new_personal_record:
            new_personal_record[row["CONDITION_SOURCE_VALUE"]] += 1

    records.append(deepcopy(new_personal_record))

    df = pd.DataFrame(records)
    df.drop(columns=["RECORD_DATETIME"], axis=1, inplace=True)

    return df


def condition_divide_from_pkl(cfg, mode, person_id):
    return pd.read_pickle(cfg.get_condition_file_path(mode, person_id))


def combined_preprocess(cfg, mode, sampling_strategy):
    print('Combined Preprocess Starts!')
    m_df = pd.read_csv(cfg.get_csv_path(measurement_csv, mode), encoding='CP949')
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    c_df = pd.read_csv(cfg.get_csv_path(condition_csv, mode))

    person_ids = get_person_ids(p_df)
    birth_dates = get_birth_dates(p_df)

    for person_id in person_ids:
        print('Person: ', person_id)
        birth_date = string_to_datetime(birth_dates[person_id])
        measurement_df = measure_divide(m_df, person_id, birth_date, sampling_strategy)
        condition_df = condition_divide(c_df, person_id, birth_date)

        measurement_df.set_index("TIME_FROM_BIRTH", inplace=True)
        condition_df.set_index("TIME_FROM_BIRTH", inplace=True)

        df = pd.merge(measurement_df, condition_df, left_index=True, right_index=True, how='outer')
        df = _sampling(df, 'front')
        df = _fillna(df)

        df.to_pickle(cfg.get_combined_file_path(mode, sampling_strategy, person_id))


def combined_preprocess_from_pkl(cfg, mode, sampling_strategy):
    print('Combined Preprocess From PKL Starts!')
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    person_ids = get_person_ids(p_df)

    for person_id in person_ids:
        print('Person: ', person_id)
        measurement_df = measure_divide_from_pkl(cfg, mode, sampling_strategy, person_id)
        condition_df = condition_divide_from_pkl(cfg, mode, person_id)

        measurement_df.set_index("TIME_FROM_BIRTH", inplace=True)
        condition_df.set_index("TIME_FROM_BIRTH", inplace=True)

        df = pd.merge(measurement_df, condition_df, left_index=True, right_index=True, how='outer')
        df = _sampling(df, 'front')
        df = _fillna(df)

        df.to_pickle(cfg.get_combined_file_path(mode, sampling_strategy, person_id))
