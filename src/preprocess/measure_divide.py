import pandas as pd
from src.utils import get_person_ids, get_birth_dates, days_hours_minutes, string_to_datetime
from src.constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, person_csv
from src.preprocess.utils import _exupperlowers, _sampling, _normalize, _fillna


def divide(m_df, person_id, birth_date):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    m_df = _exupperlowers(m_df)
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)

    records = []
    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        elif new_personal_record["MEASUREMENT_DATETIME"] != row["MEASUREMENT_DATETIME"]:
            records.append(new_personal_record)
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        if row["MEASUREMENT_SOURCE_VALUE"] in MEASUREMENT_SOURCE_VALUE_USES:
            new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        records.append(new_personal_record)

    return pd.DataFrame(records)


def measurement_preprocess(cfg, mode, sampling_strategy):
    m_df = pd.read_csv(cfg.get_csv_path(measurement_csv, mode), encoding='CP949')
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')

    person_ids = get_person_ids(p_df)
    birth_dates = get_birth_dates(p_df)

    for person_id in person_ids:
        print('Person: ', person_id)
        birth_date = string_to_datetime(birth_dates[person_id])
        person_resampled_df = divide(m_df, person_id, birth_date)
        columns = list(person_resampled_df.columns)
        for source in MEASUREMENT_SOURCE_VALUE_USES:
            if source not in columns:
                person_resampled_df[source] = None
        df = person_resampled_df[
            ["TIME_FROM_BIRTH"] + MEASUREMENT_SOURCE_VALUE_USES]
        from_birth_df = df["TIME_FROM_BIRTH"]
        df.drop(columns=["TIME_FROM_BIRTH"], axis=1, inplace=True)
        df = _sampling(df, sampling_strategy)
        df = _normalize(df)
        df = _fillna(df)
        df["TIME_FROM_BIRTH"] = from_birth_df
        df.to_pickle(cfg.get_sampled_file_path(mode, sampling_strategy, person_id))
