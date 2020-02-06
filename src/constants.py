person_csv = "/PERSON_NICU.csv"
outcome_cohort_csv = "/OUTCOME_COHORT.csv"
measurement_csv = "/MEASUREMENT_NICU.csv"
condition_csv = "/CONDITION_OCCURRENCE_NICU.csv"
preprocess_csv = "/FOR_PREPROCESS.csv"
output_csv = "/output.csv"

## Using 12 values out of 95 - initial try ##
# MEASUREMENT_SOURCE_VALUE_USES = ['HR', 'Temp', 'RR', 'SpO2', 'Pulse', 'T1', 'ABPd', 'ABPm', 'ABPs', 'NBPd',
#                                  'NBPm', 'NBPs']

## Using 72 values out of 95 - 2020.01.22 by SYS ##
MEASUREMENT_SOURCE_VALUE_USES = [
    'HR', 'RR', 'SpO2', 'Pulse', 'Temp', 'ABPm', 'ABPd', 'ABPs', 'NBPm', 'NBPs', 'NBPd',
    'SPO2-%', 'SPO2-R', 'Resp', 'PVC', 'ST-II', 'etCO2', 'SpO2 r', 'imCO2', 'ST-V1',
    'ST-I', 'ST-III', 'ST-aVF', 'ST-aVL', 'ST-aVR', 'awRR', 'CVPm', 'AoM', 'ST-V2',
    'ST-V3', 'ST-V4', 'ST-V5', 'ST-V6', 'SpO2T', 'T1', 'TV', 'Cdyn', 'PEEP', 'RRaw',
    'TVin', 'inO2', 'AoD', 'AoS', 'InsTi', 'MINVOL', 'MnAwP', 'PIP', 'MVin', 'PB', 'Poccl',
    'Pplat', 'MV', 'Patm', 'Ppeak', 'Rinsp', 'ST-V', 'sInsTi', 'sPEEP', 'sTV', 'sTrig',
    'sPSV', 'Rexp', 'highP', 'sAPkFl', 'sAWRR', 'sFIO2', 'sPIF', 'sMV', 'sO2', 'sRisTi',
    'PAPm', 'sSIMV']

CONDITION_SOURCE_VALUE_USES = [
    "P07.3", "P07.1", "P22.0", "P22.9", "P07.2", "Z38.0", "Z38.3", "P27.1", "P03.4", "P07.0", "Q21.1", "H35.1", "R62.0",
    "P59.0", "P05.1", "P91.2", "E03.1", "Z00.0", "P01.1", "D18.0", "Q25.0", "K40.9", "P00.0", "Z38.6", "P28.4", "R94.6",
    "P08.1", "Z13.9", "Q65.2", "P52.2", "P01.7", "H93.2", "P00.8", "P70.4", "B34.9", "P29.3", "P05.9", "E03.9", "Z26.9",
    "Q04.8", "H65.9", "J21.9", "P25.1", "U83.0", "M43.62", "D75.8", "P52.1", "E27.4", "P01.0", "Q53.2", "P76.0",
    "P02.7", "P70.0", "P52.9", "E80.7", "P02.0", "P01.2", "Q04.3", "R56.8", "P72.2", "R50.9", "P00.9", "Q69.1", "Z01.0",
    "N43.3", "R62.8", "K21.9", "J06.9", "E87.1", "Q53.1", "Q42.3", "P52.8", "R05", "N13.3", "G25.3", "P04.0", "Q87.2",
    "L05.9", "H91.9", "P92.9", "D22.9", "E03.8", "Q31.5", "J40", "P59.9", "P29.8", "J12.3", "J93.9", "P56.9", "P01.5",
    "E63.9", "N83.2", "J18.9", "Q62.0", "Q82.8", "P28.2", "K61.0", "N48.1", "E83.5", "R34  ", "Q63.2", "K40.3", "Z38.1",
    "Q89.9", "H66.9", "D64.9", "P96.8", "P01.3", "P35.1", "Q10.5", "P02.4", "E22.2", "P54.0", "J21.1", "Q33.6", "P26.9",
    "P74", "I28.8", "N94.8", "M67.4", "R21", "R25.1", "L92.8", "P78.1", "T81.3", "P61.2", "T18.9", "I47.1", "G93.8",
    "F98.2", "P26.1", "G00.2", "P02.1", "E88.9", "A49.0", "L74.3", "O31.2", "Q04.6", "K56.5", "S09.9", "I50.9", "R09.8",
    "E87.2", "B95.6", "Q66.8", "Q64.4", "P54.3", "Q67.3", "G00.8", "K92.2", "Q53.9", "J98.1", "P52.6", "P94.2", "K59.0",
    "R73.9", "I27.2", "R00.1", "R11", "P81.9", "E55.0", "N17.9", "L22", "Q10.3", "R68.1", "P90", "R04.8", "R16.2",
    "G91.8", "P91.7", "P52.3", "P61.0", "P59.8", "H90.2", "F19.0", "Q27.0", "D22.5", "R49.0", "R09.2", "R09.3", "S36.4",
    "P52.0", "K91.4"]

MEASUREMENT_SAMPLED_USES = [
    "HR", "Pulse", "ARTd", "ABPd", "ARTs", "ABPs", "ABPm", "NBPd", "NBP-D", "NBPm", "NBP-M", "NBPs", "NBP-S", "RR",
    "Resp", "SpO2T", "SpO2-%", "SpO2", "SPO2-R", "Temp",
]

MEASUREMENT_FEATURE_USES = ["PR", "BT", "IDBP", "IMBP", "ISBP", "DBP", "MBP", "SBP", "RR", "SPO2", "SPO2R"]

model_config = {
    'measure_dim': len(MEASUREMENT_FEATURE_USES),
    'con_dim': len(CONDITION_SOURCE_VALUE_USES),
    'embedd_dim': 64,
    'drop_prob': 0.1,
    'num_heads': 4,
    'ffn_dim': 256,
    'num_labels': 1,
    'num_layers': 2,
    'num_stacks': 2,
    'hidden_dim': 64
}

hyperparams = {
    'batch_size': 256,
    'lr': 0.0001,
    'weight_decay': 0.01,
    'sampling_strategy': 'front',
    'max_seq_len': 1024,
    'epochs': 20,
    'gamma': 2.0,
    'alpha': 0.25,
    'prior_prob': 0.059,
    'warmup_proportion': 0.1
}
