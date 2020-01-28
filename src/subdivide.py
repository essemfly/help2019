import pandas as pd
from .constants import MEASUREMENT_SOURCE_VALUE_USES
from .utils import days_hours_minutes


def divide(m_df, person_id, birth_date):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)
    divided_df = pd.DataFrame()

    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        elif new_personal_record["MEASUREMENT_DATETIME"] != row["MEASUREMENT_DATETIME"]:
            divided_df = divided_df.append(new_personal_record, ignore_index=True)
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"],
                                   "TIME_FROM_BIRTH": days_hours_minutes(row["MEASUREMENT_DATETIME"] - birth_date)}

        if row["MEASUREMENT_SOURCE_VALUE"] in MEASUREMENT_SOURCE_VALUE_USES:
            new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        divided_df = divided_df.append(new_personal_record, ignore_index=True)

    return divided_df
