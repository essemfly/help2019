import numpy as np
from datetime import datetime, timedelta
from .constants import MEASUREMENT_SOURCE_VALUE_USES

## Predefined Features Setting ##

np.seterr(divide='ignore', invalid='ignore')

USE_DURATION = True
USE_AVERAGE = True
USE_MEDIAN = False
USE_MODE = False
USE_MINMAX = True
USE_STD = True
USE_COUNT = True
USE_FREQ = True
USE_LAST_12_AVG = True
USE_LAST_6_AVG = True
USE_LAST_2_AVG = True
USE_LAST_VALUE = True

##

def string_to_datetime_m(time_str, format='%Y-%m-%d %H:%M'):
    return datetime.strptime(time_str[:16], format)

def datetime_to_string(_datetime, format='%Y-%m-%d %H:%M'):
    return _datetime.strftime(format)

def days_hours_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60

def extract_df(measurement_df, outcome_df, column_list = MEASUREMENT_SOURCE_VALUE_USES):
    df = measurement_df
    odf = outcome_df
    for index, row in outcome_df.iterrows():
        pid = row['SUBJECT_ID']
        starttime = row['COHORT_START_DATE']
        endtime = row['COHORT_END_DATE']
        sub_m_df = df[(df['PERSON_ID'] == pid) & (df['MEASUREMENT_DATETIME'] >= starttime) & (df['MEASUREMENT_DATETIME'] <= endtime)]
        
        cohort_start_date = string_to_datetime_m(starttime)
        cohort_end_date = string_to_datetime_m(endtime)
        cohort_mins = cohort_end_date - cohort_start_date
        minutes = days_hours_minutes(cohort_mins)
        
        if (USE_DURATION):
            odf.at[index, 'duration'] = minutes
        
        for i in range(len(MEASUREMENT_SOURCE_VALUE_USES)):
            selected_m_df = sub_m_df[sub_m_df['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]]
            if (USE_AVERAGE):
                odf.at[index, 'avg' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].mean()
            if (USE_MEDIAN):
                odf.at[index, 'med' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].median()
            if (USE_MODE):
                odf.at[index, 'mod' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].mode()
            if (USE_MINMAX):
                odf.at[index, 'min' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].min()
                odf.at[index, 'max' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].max()
            if (USE_STD):
                odf.at[index, 'std' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].std()
            if (USE_COUNT):
                odf.at[index, 'count' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].count()
            if (USE_FREQ): 
                odf.at[index, 'freq' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df['VALUE_SOURCE_VALUE'].count()/minutes

            last12_m_df = selected_m_df[selected_m_df['MEASUREMENT_DATETIME'] > datetime_to_string(cohort_end_date - timedelta(hours=12))]
            last6_m_df = selected_m_df[selected_m_df['MEASUREMENT_DATETIME'] > datetime_to_string(cohort_end_date - timedelta(hours=6))]
            last2_m_df = selected_m_df[selected_m_df['MEASUREMENT_DATETIME'] > datetime_to_string(cohort_end_date - timedelta(hours=2))]
            if (USE_LAST_12_AVG):
                odf.at[index, 'avg12' + MEASUREMENT_SOURCE_VALUE_USES[i]] = last12_m_df['VALUE_SOURCE_VALUE'].mean() if last12_m_df['VALUE_SOURCE_VALUE'].shape[0]>0 else np.nan
            if (USE_LAST_6_AVG):
                odf.at[index, 'avg6' + MEASUREMENT_SOURCE_VALUE_USES[i]] = last6_m_df['VALUE_SOURCE_VALUE'].mean() if last6_m_df['VALUE_SOURCE_VALUE'].shape[0]>0 else np.nan
            if (USE_LAST_2_AVG):
                odf.at[index, 'avg2' + MEASUREMENT_SOURCE_VALUE_USES[i]] = last2_m_df['VALUE_SOURCE_VALUE'].mean() if last2_m_df['VALUE_SOURCE_VALUE'].shape[0]>0 else np.nan
            if (USE_LAST_VALUE):
                odf.at[index, 'lastval' + MEASUREMENT_SOURCE_VALUE_USES[i]] = selected_m_df.iloc[-1]['VALUE_SOURCE_VALUE'] if selected_m_df.iloc[-1:]['VALUE_SOURCE_VALUE'].shape[0]>0 else np.nan

    return odf.iloc[:,5:]