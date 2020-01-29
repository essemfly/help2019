import os
import pandas as pd
from tensorboardX import SummaryWriter
from datetime import date
from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .resample import resample_and_save_by_user
from .subdivide import subdivide
from .utils import get_start_end, get_person_ids, get_birth_date

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def train(cfg, writer):
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')
    pass


def main(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    print("Train function runs")

    # resample_and_save_by_user(cfg, writer)
    subdivide(cfg, writer)
    # train(cfg, writer)
