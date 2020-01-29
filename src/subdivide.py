import pandas as pd
from .utils import get_person_ids, days_hours_minutes, get_birth_date
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, person_csv
from datetime import datetime


def divide(m_df, person_id, birth_date):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
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


def subdivide(cfg, writer):
    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')

    person_ids = get_person_ids(p_df)

    for person_id in person_ids:
        print('Person: ', person_id)
        birth_date = get_birth_date(p_df, person_id)
        try:
            person_resampled_df = divide(m_df, person_id, birth_date)
            person_resampled_df.to_csv(f'{cfg.VOLUME_DIR}/{str(person_id)}.csv')
        except ValueError as e:
            print('e', e)
