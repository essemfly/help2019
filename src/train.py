import os
import pandas as pd
from tensorboardX import SummaryWriter
from datetime import date
from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .resample import resample

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def resample_and_save_by_user(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    person_ids = get_person_ids(cfg)

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')

    for person_id in person_ids:
        print('USER_ID: ', person_id)
        df = resample(m_df, o_df, person_id, column_list=MEASUREMENT_SOURCE_VALUE_USES)
        df.to_csv(cfg.VOLUME_DIR + "/" + str(person_id) + "_measurement.csv")

    # FOR TEST IN TENSORBOARD
    person_id = person_ids[0]
    df = resample(m_df, o_df, person_id, column_list=MEASUREMENT_SOURCE_VALUE_USES)
    idx = 0
    for index, row in df.iterrows():
        idx += 1
        for source in MEASUREMENT_SOURCE_VALUE_USES:
            writer.add_scalar(str(person_id) + "-" + source, row[source], idx)

    writer.close()


def train(env):
    print("Train function runs")
    # resample_and_save_by_user(env)


def get_person_ids(cfg):
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')
    return p_df.loc[:, "PERSON_ID"].values.tolist()
