from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.constants import MEASUREMENT_SOURCE_VALUE_USES, CONDITION_SOURCE_VALUE_USES, measurement_csv, person_csv, \
    condition_csv
from src.utils import get_person_ids, datetime_to_string, string_to_datetime


def measurement_preprocess(cfg, mode):
    print("Sample by Hours start!")
    m_df = pd.read_csv(cfg.get_csv_path(measurement_csv, mode), encoding='CP949')
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)

    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    c_df = pd.read_csv(cfg.get_csv_path(condition_csv, mode))

    person_ids = get_person_ids(p_df)

    for _, person_id in enumerate(tqdm(person_ids, desc='People iteration')):
        df = m_df[m_df['PERSON_ID'] == person_id]
        first_record = df.iloc[0]["MEASUREMENT_DATETIME"]
        first_hour = string_to_datetime(datetime_to_string(first_record, format='%Y-%m-%d %H'), format='%Y-%m-%d %H')

        records = []
        next_hour = first_hour + timedelta(hours=1)
        in_hour_measures = {'MEASUREMENT_DATETIME': next_hour}

        for _, row in df.iterrows():
            if row["MEASUREMENT_DATETIME"] < next_hour:
                if row["MEASUREMENT_SOURCE_VALUE"] in in_hour_measures:
                    in_hour_measures[row["MEASUREMENT_SOURCE_VALUE"]].append(row["VALUE_SOURCE_VALUE"])
                else:
                    in_hour_measures[row["MEASUREMENT_SOURCE_VALUE"]] = [row["VALUE_SOURCE_VALUE"]]
            else:
                for key in in_hour_measures:
                    if key == 'MEASUREMENT_DATETIME':
                        continue
                    in_hour_measures[key] = np.mean(in_hour_measures[key])
                records.append(in_hour_measures)
                next_hour = next_hour + timedelta(hours=1)
                in_hour_measures = {'MEASUREMENT_DATETIME': next_hour,
                                    row["MEASUREMENT_SOURCE_VALUE"]: [row["VALUE_SOURCE_VALUE"]]}

        record_df = pd.DataFrame(records)
        print(f'Describe {person_id} Measure df', record_df.describe())
        record_df.to_pickle(cfg.get_hourly_divided_measure_file_path(mode, person_id))
