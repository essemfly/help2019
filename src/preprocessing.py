import pandas as pd
from .constants import MEASUREMENT_SOURCE_VALUE_USES, preprocess_csv

def exupperlowers (measurement_df, column_list=MEASUREMENT_SOURCE_VALUE_USES):
    ref_m = pd.read_csv('sample' + preprocess_csv, encoding='CP949')
    df = measurement_df
    for i in range(len(MEASUREMENT_SOURCE_VALUE_USES)):
        if (ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['METHODS'] == 1):
            lower = ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['LOWER']
            upper = ref_m[ref_m['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]].iloc[0]['UPPER']
            df = df.drop(df[(df['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]) & (df['VALUE_SOURCE_VALUE'] <= lower)].index)
            df = df.drop(df[(df['MEASUREMENT_SOURCE_VALUE'] == MEASUREMENT_SOURCE_VALUE_USES[i]) & (df['VALUE_SOURCE_VALUE'] > upper)].index)
    return df.reset_index()