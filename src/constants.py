person_csv = "/PERSON_NICU.csv"
outcome_cohort_csv = "/OUTCOME_COHORT.csv"
measurement_csv = "/MEASUREMENT_NICU.csv"
condition_csv = "/CONDITION_OCCURENCE_NICU.csv"
preprocess_csv = "/FOR_PREPROCESS.csv"

## Using 12 values out of 95 - initial try ##
# MEASUREMENT_SOURCE_VALUE_USES = ['HR', 'Temp', 'RR', 'SpO2', 'Pulse', 'T1', 'ABPd', 'ABPm', 'ABPs', 'NBPd',
#                                  'NBPm', 'NBPs']

## Using 72 values out of 95 - 2020.01.22 by SYS ##
MEASUREMENT_SOURCE_VALUE_USES = ['HR', 'RR', 'SpO2', 'Pulse', 'Temp', 'ABPm', 'ABPd', 'ABPs', 'NBPm', 'NBPs', 'NBPd',
                                 'SPO2-%', 'SPO2-R', 'Resp', 'PVC', 'ST-II', 'etCO2', 'SpO2 r', 'imCO2', 'ST-V1',
                                 'ST-I', 'ST-III', 'ST-aVF', 'ST-aVL', 'ST-aVR', 'awRR', 'CVPm', 'AoM', 'ST-V2',
                                 'ST-V3', 'ST-V4', 'ST-V5', 'ST-V6', 'SpO2T', 'T1', 'TV', 'Cdyn', 'PEEP', 'RRaw',
                                 'TVin', 'inO2', 'AoD', 'AoS', 'InsTi', 'MINVOL', 'MnAwP', 'PIP', 'MVin', 'PB', 'Poccl',
                                 'Pplat', 'MV', 'Patm', 'Ppeak', 'Rinsp', 'ST-V', 'sInsTi', 'sPEEP', 'sTV', 'sTrig',
                                 'sPSV', 'Rexp', 'highP', 'sAPkFl', 'sAWRR', 'sFIO2', 'sPIF', 'sMV', 'sO2', 'sRisTi',
                                 'PAPm', 'sSIMV']
