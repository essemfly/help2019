from datetime import datetime, timedelta

import pandas as pd

from .constants import person_csv


def string_to_datetime(time_str, format='%Y-%m-%d %H:%M'):
    return datetime.strptime(time_str[:16], format)


def datetime_to_string(_datetime, format='%Y-%m-%d %H:%M'):
    return _datetime.strftime(format)


def round_time(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None: dt = datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def ceil_time(dt=None):
    """올림으로
    """
    if dt is None:
        dt = datetime.now()
    if dt.second > 0:
        return dt.replace(tzinfo=None, second=0, microsecond=0) + timedelta(minutes=1)
    else:
        return dt.replace(tzinfo=None, second=0, microsecond=0)


def days_hours_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60


def isNaN(num):
    return num != num


def sort_by_datetime(m_df):
    m_df["MEASUREMENT_DATETIME"] = pd.to_datetime(m_df["MEASUREMENT_DATETIME"], format="%Y-%m-%d %H:%M:%S")
    m_df.sort_values("MEASUREMENT_DATETIME")
    return m_df


def get_start_end(o_df, subject_id):
    person_df = o_df[o_df['SUBJECT_ID'] == subject_id]
    person_df = person_df.sort_values("COHORT_START_DATE")
    cohort_start_date = person_df.iloc[0]["COHORT_START_DATE"]
    cohort_start_date = string_to_datetime(cohort_start_date)

    person_df = person_df.sort_values("COHORT_END_DATE")
    cohort_end_date = person_df.iloc[-1]["COHORT_END_DATE"]
    cohort_end_date = string_to_datetime(cohort_end_date)
    return cohort_start_date, cohort_end_date


def get_person_ids(p_df):
    return p_df.loc[:, "PERSON_ID"].values.tolist()


def get_birth_dates(p_df):
    return p_df[["PERSON_ID", "BIRTH_DATETIME"]].set_index("PERSON_ID").to_dict(orient='dict')[
        "BIRTH_DATETIME"]
