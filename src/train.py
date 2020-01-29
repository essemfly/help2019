import os
import pandas as pd
from tensorboardX import SummaryWriter
from datetime import date
from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .resample import resample
from .subdivide import divide
from .utils import get_start_end, get_person_ids, get_birth_date
from .preprocessing import exupperlowers

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def resample_and_save_by_user(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
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


def subdivide(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    m_df = exupperlowers(m_df)
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')

    person_ids = get_person_ids(p_df)

    for person_id in person_ids:
        try:
            birth_date = get_birth_date(p_df, person_id)
            person_resampled_df = divide(m_df, person_id, birth_date)
            person_resampled_df.to_csv(f'{cfg.VOLUME_DIR}/{str(person_id)}.csv')
        except ValueError as e:
            print('e', e)


def train(env):
    print("Train function runs")
    # resample_and_save_by_user(env)
    subdivide(env)
