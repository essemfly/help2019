from copy import deepcopy
import pandas as pd
from .constants import CONDITION_SOURCE_VALUE_USES, person_csv, condition_csv
from .utils import get_person_ids, get_birth_dates, string_to_datetime, days_hours_minutes


def condition_preprocess(cfg, mode):
    print('Condition Preprocess Starts!')
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')

    person_ids = get_person_ids(p_df)
    birth_dates = get_birth_dates(p_df)

    for person_id in person_ids:
        c_df = pd.read_csv(cfg.get_csv_path(condition_csv, mode))
        birth_date = string_to_datetime(birth_dates[person_id])
        c_df = c_df[c_df["PERSON_ID"] == person_id]
        c_df.loc[:, "CONDITION_START_DATETIME"] = pd.to_datetime(c_df["CONDITION_START_DATETIME"],
                                                                 format="%Y-%m-%d %H:%M")
        c_df.sort_values("CONDITION_START_DATETIME", inplace=True)

        print('Counts:', f'{person_id}: {len(c_df)}')
        records = []
        new_personal_record = {condition: 0 for condition in CONDITION_SOURCE_VALUE_USES}
        new_personal_record["CONDITION_DATETIME"] = birth_date
        new_personal_record["TIME_FROM_BIRTH"] = 0
        for idx, row in c_df.iterrows():
            if "CONDITION_DATETIME" not in new_personal_record:
                new_personal_record["CONDITION_DATETIME"] = row["CONDITION_START_DATETIME"]
                new_personal_record["TIME_FROM_BIRTH"] = days_hours_minutes(
                    row["CONDITION_START_DATETIME"] - birth_date)
            elif new_personal_record["CONDITION_DATETIME"] != row["CONDITION_START_DATETIME"]:
                if row["CONDITION_SOURCE_VALUE"] in new_personal_record:
                    records.append(deepcopy(new_personal_record))
                    new_personal_record["CONDITION_DATETIME"] = row["CONDITION_START_DATETIME"]
                    new_personal_record["TIME_FROM_BIRTH"] = days_hours_minutes(
                        row["CONDITION_START_DATETIME"] - birth_date)

            if row["CONDITION_SOURCE_VALUE"] in new_personal_record:
                new_personal_record[row["CONDITION_SOURCE_VALUE"]] += 1

        records.append(deepcopy(new_personal_record))

        df = pd.DataFrame(records)
        df.drop(columns=["CONDITION_DATETIME"], axis=1, inplace=True)
        df.to_pickle(cfg.get_condition_file_path(mode, person_id))
