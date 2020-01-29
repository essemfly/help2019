import pandas as pd
import os
from datetime import timedelta
from .utils import get_start_end, get_person_ids
from .preprocessing import exupperlowers
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv
from .utils import days_hours_minutes, ceil_time, isNaN, datetime_to_string, \
    string_to_datetime


def get_impute_strategy(strategy):
    if strategy.lower() == 'zero':
        return 0
    elif strategy.lower() == 'nan':
        return float('nan')
    else:
        print('Error: Please check the impute_strategy (resample func)')
        raise RuntimeError('Invalid impute strategy')


def sampling_strategy(df, new_df, strategy):
    for index, row in df.iterrows():
        row_time = row['MEASUREMENT_DATETIME']
        col_type = row['MEASUREMENT_SOURCE_VALUE']
        value = row['VALUE_SOURCE_VALUE']  # VALUE_AS_NUMBER

        if strategy == 'ignore':
            continue
        elif strategy == 'nearest':
            continue
        elif strategy == 'average':
            continue
        else:
            print("Error : Check the sampling strategy! Check resample func.")
            raise RuntimeError('Invalid sampling strategy')
    pass


def resample(measure_df, subject_id, start_datetime, end_datetime, unit_min=1, impute_strategy='nan',
             sampling_strategy='ignore'):
    filtered_df = measure_df[measure_df['PERSON_ID'] == subject_id]
    cohort_start_date, cohort_end_date = start_datetime, end_datetime
    cohort_mins = cohort_end_date - cohort_start_date
    minutes = days_hours_minutes(cohort_mins)

    # 큰 empty dataframe을 만들고 넣는 방식으로. # edited by DWLee, 2020.01.12.
    # 첫번째 column을 datetime type으로 만들기 위해 index list생성.
    if unit_min < 1:
        print('Error: Please check the unit_min. Condition: unit_min>=1 (resample func)')
        return
    index_list_chomchom = []
    index_list = []

    for idx in range(0, minutes + 1):  # ,unit_min):
        min2date = timedelta(minutes=idx)
        cur_date = datetime_to_string(cohort_start_date + min2date)
        index_list_chomchom.append(cur_date)
        if (idx % unit_min) == 0:
            index_list.append(cur_date)

    # Generates empty dataframe.
    # rows : total (minutes+1) number of rows : datetime type
    # cols : [HR, Temp, ..., Pulse]

    df = pd.DataFrame(index=index_list, columns=MEASUREMENT_SOURCE_VALUE_USES)
    df_chomchom = pd.DataFrame(index=index_list_chomchom, columns=MEASUREMENT_SOURCE_VALUE_USES)

    fill_what_for_sampling_strategy = get_impute_strategy(impute_strategy)

    df = df.fillna(fill_what_for_sampling_strategy)  # with 'init value'
    df_chomchom = df_chomchom.fillna(fill_what_for_sampling_strategy)  # with 'init value'

    ############################################################################
    if sampling_strategy.lower() == 'ignore':
        for index, row in filtered_df.iterrows():
            row_time = row['MEASUREMENT_DATETIME']
            col_type = row['MEASUREMENT_SOURCE_VALUE']
            value = row['VALUE_SOURCE_VALUE']  # VALUE_AS_NUMBER']
            try:
                df_chomchom.loc[row_time][col_type] = value
            except KeyError:
                # Ignore sub-min time entries
                continue

    elif sampling_strategy.lower() == 'nearest':
        for index, row in filtered_df.iterrows():
            row_time = row['MEASUREMENT_DATETIME']
            col_type = row['MEASUREMENT_SOURCE_VALUE']
            value = row['VALUE_SOURCE_VALUE']  # VALUE_AS_NUMBER']
            try:
                new_row_time_ = ceil_time(string_to_datetime(row_time))
                if new_row_time_ > cohort_end_date:
                    continue
                new_row_time = datetime_to_string(new_row_time_)
                df_chomchom.loc[new_row_time][col_type] = value
            except KeyError:
                print("Exception! This should NOT Happen!!! check resample func. Key:%s" % new_row_time)
                print("TIME : %s --> %s" % (row_time, new_row_time))
                continue

        ## Sampling strategy가 nearest일 경우 필요한 init value.
        cur_nearest = {x: fill_what_for_sampling_strategy for x in MEASUREMENT_SOURCE_VALUE_USES}
        for index, row in df_chomchom.iterrows():
            try:
                for a_col in MEASUREMENT_SOURCE_VALUE_USES:
                    cur_nearest[a_col] = row[a_col]
                    df.loc[index][a_col] = cur_nearest[a_col]
            except KeyError:
                # 빈칸: 넘어가면 알아서 nearest가 채워짐.
                continue


    elif sampling_strategy.lower() == 'average':
        ## Sampling strategy가 average일 경우 필요한 init value.
        # cur_nearest = {x:fill_what_for_sampling_strategy for x in column_list }
        for index, row in filtered_df.iterrows():
            row_time = row['MEASUREMENT_DATETIME']
            col_type = row['MEASUREMENT_SOURCE_VALUE']
            value = row['VALUE_SOURCE_VALUE']  # VALUE_AS_NUMBER']
            try:
                new_row_time = datetime_to_string(ceil_time(string_to_datetime(row_time)))
                if impute_strategy.lower() == 'nan' and isNaN(df_chomchom.loc[new_row_time][col_type]):
                    df_chomchom.loc[new_row_time][col_type] = value
                elif impute_strategy.lower() == 'zero' and df_chomchom.loc[new_row_time][col_type] == 0:
                    df_chomchom.loc[new_row_time][col_type] = value
                else:
                    prev_inputed_value = df_chomchom.loc[new_row_time][col_type]
                    # should be revised more precisely
                    df_chomchom.loc[new_row_time][col_type] = (value + prev_inputed_value) / 2.
            except(KeyError):
                # 없을텐디
                print("Exception! This should NOT Happen!! average. check resample func. Key:%s" % new_row_time)
                continue
    else:
        print("Error : Check the sampling strategy! Check resample func.")
        return

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


''' 
MAIN FUNCTION
'''


def resample_and_save_by_user(cfg, writer):
    person_ids = get_person_ids(cfg)

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')

    m_df = exupperlowers(m_df)

    unit_min = 1
    impute_strategy = 'nan'
    sampling_strategy = 'ignore'

    for person_id in person_ids:
        print('USER_ID: ', person_id)
        start_datetime, end_datetime = get_start_end(o_df, person_id)
        df = resample(m_df, person_id, start_datetime, end_datetime, unit_min=unit_min, impute_strategy=impute_strategy,
                      sampling_strategy=sampling_strategy)
        os.makedirs(f'{cfg.VOLUME_DIR}/{sampling_strategy}_{impute_strategy}_{unit_min}/', exist_ok=True)
        df.to_csv(f'{cfg.VOLUME_DIR}/{sampling_strategy}_{impute_strategy}_{unit_min}/{str(person_id)}.csv')

    # FOR TEST IN TENSORBOARD
    person_id = person_ids[0]
    df = resample(m_df, o_df, person_id)
    idx = 0
    for index, row in df.iterrows():
        idx += 1
        for source in MEASUREMENT_SOURCE_VALUE_USES:
            writer.add_scalar(str(person_id) + "-" + source, row[source], idx)

    writer.close()
