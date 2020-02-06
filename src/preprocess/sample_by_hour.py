from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, person_csv, \
    condition_csv
from src.utils import get_person_ids, datetime_to_string, string_to_datetime
from src.preprocess.utils import _sampling, _normalize, _fillna, _exupperlowers


def measure11_dfs(cfg, mode):
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    person_ids = get_person_ids(p_df)
    hourly_sampled_dfs = {}

    for person_id in person_ids:
        hourly_sampled_dfs[person_id] = pd.read_pickle(cfg.get_hourly_divided_11measures_df_path(mode, person_id))
    return hourly_sampled_dfs


def measure72_dfs(cfg, mode):
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    person_ids = get_person_ids(p_df)
    hourly_sampled_dfs = {}

    for person_id in person_ids:
        hourly_sampled_dfs[person_id] = pd.read_pickle(cfg.get_hourly_divided_11measures_df_path(mode, person_id))
    return hourly_sampled_dfs


def convert_features_to_dataset(cfg, mode):
    print("72 Measures are Saved!")
    p_df = pd.read_csv(cfg.get_csv_path(person_csv, mode), encoding='CP949')
    person_ids = get_person_ids(p_df)

    for _, person_id in enumerate(tqdm(person_ids, desc='People iteration')):
        df = pd.read_pickle(cfg.get_hourly_divided_measure_file_path(mode, person_id))
        columns = df.columns.tolist()

        nan_value = float("NaN")
        for required_column in MEASUREMENT_SOURCE_VALUE_USES:
            if required_column not in columns:
                df[required_column] = nan_value

        measure_time = df["MEASUREMENT_DATETIME"]
        df.drop(columns=["MEASUREMENT_DATETIME"], axis=1, inplace=True)

        df = _sampling(df, 'front')
        df = _normalize(df)
        df = _fillna(df)
        df["MEASUREMENT_DATETIME"] = measure_time
        df = df[
            ["MEASUREMENT_DATETIME"] + MEASUREMENT_SOURCE_VALUE_USES]
        # new_df["PR"] = df[["HR", "Pulse"]].mean(axis=1)
        # new_df["BT"] = df["Temp"]
        # new_df["IDBP"] = df[["ARTd", "ABPd"]].mean(axis=1)
        # new_df["IMBP"] = df["ABPm"]
        # new_df["ISBP"] = df[["ARTs", "ABPs"]].mean(axis=1)
        # new_df["DBP"] = df[["NBPd", "NBP-D"]].mean(axis=1)
        # new_df["MBP"] = df[["NBPm", "NBP-M"]].mean(axis=1)
        # new_df["SBP"] = df[["NBPs", "NBP-S"]].mean(axis=1)
        # new_df["RR"] = df[["RR", "Resp"]].mean(axis=1)
        # new_df["SPO2"] = df[["SpO2T", "SpO2-%", "SpO2"]].mean(axis=1)
        # new_df["SPO2R"] = df["SPO2-R"]

        df.to_pickle(cfg.get_hourly_divided_72measures_df_path(mode, person_id))


def measurement_preprocess(cfg, mode):
    print("Sample by Hours start!")
    m_df = pd.read_csv(cfg.get_csv_path(measurement_csv, mode), encoding='CP949')
    m_df.loc[:, "MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M")
    m_df.sort_values("MEASUREMENT_DATETIME", inplace=True)
    m_df = _exupperlowers(m_df)

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
        record_df.to_pickle(cfg.get_hourly_divided_measure_file_path(mode, person_id))
