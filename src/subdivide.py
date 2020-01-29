import pandas as pd
from .utils import get_person_ids
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, person_csv


def divide(m_df, person_id):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    divided_df = pd.DataFrame()

    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}

        elif new_personal_record["MEASUREMENT_DATETIME"] != row["MEASUREMENT_DATETIME"]:
            divided_df = divided_df.append(new_personal_record, ignore_index=True)
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}

        if row["MEASUREMENT_SOURCE_VALUE"] in MEASUREMENT_SOURCE_VALUE_USES:
            new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        divided_df = divided_df.append(new_personal_record, ignore_index=True)

    return divided_df


def subdivide(cfg, writer):
    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')

    person_ids = get_person_ids(p_df)

    for person_id in person_ids:
        print('Person: ', person_id)
        try:
            person_resampled_df = divide(m_df, person_id)
            person_resampled_df.to_csv(f'{cfg.VOLUME_DIR}/{str(person_id)}.csv')
        except ValueError as e:
            print('e', e)
