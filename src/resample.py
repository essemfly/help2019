from datetime import datetime, timedelta
import pandas as pd
from .constants import MEASUREMENT_SOURCE_VALUE_USES


def string_to_datetime(time_str, format='%Y-%m-%d %H:%M:%S'):
    # '%Y-%m-%d %H:%M:%S %p'
    # edited by DWLee 2020.01.12. : Some dates have 'seconds' strings
    return datetime.strptime(time_str, format)


def datetime_to_string(_datetime, format='%Y-%m-%d %H:%M:%S'):
    # '%Y-%m-%d %H:%M:%S %p'
    # edited by DWLee 2020.01.12. : Some dates have 'seconds' strings
    return _datetime.strftime(format)


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None: dt = datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def days_hours_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60


def isNaN(num):
    return num != num


def resample(measure_df, outcome_df, subject_id, unit_min=1, impute_strategy='nan', sampling_strategy='ignore',
             column_list=MEASUREMENT_SOURCE_VALUE_USES):
    filtered_df = measure_df[measure_df['PERSON_ID'] == subject_id]
    cohort_start_date = outcome_df[outcome_df['SUBJECT_ID'] == subject_id].iloc[0]["COHORT_START_DATE"]
    cohort_start_date = string_to_datetime(cohort_start_date)
    cohort_end_date = outcome_df[outcome_df['SUBJECT_ID'] == subject_id].iloc[-1]["COHORT_END_DATE"]
    cohort_end_date = string_to_datetime(cohort_end_date)
    cohort_mins = cohort_end_date - cohort_start_date
    minutes = days_hours_minutes(cohort_mins)

    index_list = []
    # print(minutes)
    for idx in range(0, minutes + 1, unit_min):
        min2date = timedelta(minutes=idx)
        cur_date = datetime_to_string(cohort_start_date + min2date)
        index_list.append(cur_date)

    # Generates empty dataframe.
    # rows : total (minutes+1) number of rows : datetime type
    # cols : [HR, Temp, ..., Pulse]
    df = pd.DataFrame(index=index_list, columns=column_list)
    if impute_strategy.lower() == 'zero':
        df = df.fillna(0)  # with 0s rather than NaNs
    elif impute_strategy.lower() == 'nan':
        df = df.fillna(float('nan'))  # with 0s rather than NaNs
    else:
        print('Error: Please check the impute_strategy (resample func)')
        return
    # else: default of DataFrame init is NaN

    for index, row in filtered_df.iterrows():
        row_time = row['MEASUREMENT_DATETIME']
        col_type = row['MEASUREMENT_SOURCE_VALUE']
        value = row['VALUE_SOURCE_VALUE']  # VALUE_AS_NUMBER']

        try:
            df.loc[row_time][col_type] = value
        except(KeyError):
            # This happens the df has no index for filtered_df - datetime

            if sampling_strategy.lower() == 'ignore':
                continue
            # elif sampling_strategy.lower()=='average':
            # elif sampling_strategy.lower()=='nearest':
            new_row_time = datetime_to_string(roundTime(string_to_datetime(row_time)))

    # dataframe for label data
    # df_label = pd.DataFrame(index=index_list, columns=['LABEL'])
    # df_label = df_label.fillna(float('nan'))
    # sub_df = outcome_df[outcome_df['SUBJECT_ID'] == subject_id]
    # for _, row in sub_df.iterrows():
    #     row_time = row["COHORT_END_DATE"]
    #     df_label.loc[row_time]['LABEL'] = int(row["LABEL"])
    # cur_label = 0  # 0: There is no critical event.
    # for index, row in df_label.iterrows():
    #     row_label = row["LABEL"]
    #     if isNaN(row_label):
    #         row["LABEL"] = cur_label
    #     else:
    #         cur_label = int(row_label)
    return df
