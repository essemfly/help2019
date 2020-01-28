import pandas as pd
from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, person_csv
from .models import PersonalRecord
from .utils import get_birth_date, get_person_ids


def divide(m_df, person_id, birth_date, unit_min=1, sampling_strategy='ignore'):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    m_df["MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME")
    # m_df["TIME_FROM_BIRTH"] = m_df[M_DATETIME] - birth_date
    # resampled_df = pd.DataFrame(columns=column_list)
    # column_list = ["MEASUREMENT_DATETIME", "TIME_FROM_BIRTH"] + MEASUREMENT_SOURCE_VALUE_USES

    records = []
    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}
        elif new_personal_record.measurement_datetime != row["MEASUREMENT_DATETIME"]:
            records.append(new_personal_record)
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}

        new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        records.append(new_personal_record)

    # TODO: NOT IMPLEMENTED YET
    new_df = pd.DataFrame.from_dict(records, orient='columns')

    return new_df


def main(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')

    person_ids = get_person_ids(cfg)

    sampling_strategy = 'ignore'
    unit_min = 1

    for person_id in person_ids:
        birth_date = get_birth_date(p_df, person_id)
        person_resampled_df = divide(m_df, person_id, birth_date, unit_min=unit_min,
                                     sampling_strategy=sampling_strategy)
        person_resampled_df.to_csv(f'{cfg.VOLUME_DIR}/{str(person_id)}.csv')
