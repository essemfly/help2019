import argparse
import os
import pandas as pd
from tensorboardX import SummaryWriter
from .config import LocalConfig, ProdConfig
from .constants import measurement_csv, outcome_cohort_csv
from .resample import resample

ID = os.environ.get('ID', '20200120')


def train(env):
    MEASUREMENT_SOURCE_VALUE_USES = ['HR', 'Temp', 'RR', 'SpO2', 'Pulse']
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv)
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv)

    person_id = 47715145949628715
    df, _ = resample(m_df, o_df, person_id, column_list=MEASUREMENT_SOURCE_VALUE_USES)

    idx = 0
    for index, row in df.iterrows():
        idx += 1
        for source in MEASUREMENT_SOURCE_VALUE_USES:
            writer.add_scalar(source, row[source], idx)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    args = parser.parse_args()
    print('ENVIRONMENT : ', args.env)
    train(args.env)
