import argparse
import os
import pandas as pd
from tensorboardX import SummaryWriter
from .config import LocalConfig, ProdConfig
from .constants import person_csv, measurement_csv

ID = os.environ.get('ID', '20200120')


def train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv)
    hr_df = m_df[m_df['MEASUREMENT_SOURCE_VALUE'] == "HR"]

    X = hr_df["MEASUREMENT_DATETIME"]
    Y = hr_df["VALUE_SOURCE_VALUE"]

    idx = 0
    for x, y in zip(X, Y):
        idx += 1
        writer.add_scalar('', y, idx)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training start')
    parser.add_argument('env', metavar='ENV', type=str, default='localhost')
    args = parser.parse_args()
    print('ENVIRONMENT : ', args.env)
    train(args.env)
