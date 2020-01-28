import pandas as pd


def divide(m_df, person_id):
    m_df = m_df[m_df['PERSON_ID'] == person_id]
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)
    divided_df = pd.DataFrame()

    new_personal_record = None
    for idx, row in m_df.iterrows():
        if new_personal_record is None:
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}
        elif new_personal_record["MEASUREMENT_DATETIME"] != row["MEASUREMENT_DATETIME"]:
            divided_df = divided_df.append(new_personal_record, ignore_index=True)
            new_personal_record = {"MEASUREMENT_DATETIME": row["MEASUREMENT_DATETIME"]}

        new_personal_record[row["MEASUREMENT_SOURCE_VALUE"]] = row["VALUE_SOURCE_VALUE"]

    if new_personal_record is not None:
        divided_df = divided_df.append(new_personal_record, ignore_index=True)

    return divided_df
