import os
import pandas as pd
from tensorboardX import SummaryWriter
from datetime import date
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.python.client import device_lib 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .resample import resample
from .preprocessing import exupperlowers
from .featureextraction import extract_df

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))


def resample_and_save_by_user(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, ID))
    person_ids = get_person_ids(cfg)

    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')
    
    m_df = exupperlowers(m_df)  ## preprocessing by excluding predefined outliers - 200127 by SYS

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
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print("Train function runs")
    print(device_lib.list_local_devices())
    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    m_df = exupperlowers(m_df)  ## preprocessing by excluding predefined outliers - 200127 by SYS
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')
    feature_X = extract_df(m_df, o_df, column_list=MEASUREMENT_SOURCE_VALUE_USES)
    sc = StandardScaler()
    X = sc.fit_transform(feature_X.values)
    Y = o_df['LABEL'].values

    classifier = Sequential()
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = feature_X.shape[1]))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #classifier.compile(optimizer = 'adam', loss=['binary_crossentropy'], metrics = ['accuracy'])
    classifier.compile(optimizer = 'adam', loss=[focal_loss(gamma=2.,alpha=.25)], metrics = ['accuracy'])
    classifier.fit(X, Y, batch_size = 20, epochs = 1)
    classifier.save(cfg.LOG_DIR + '/' + str(ID) +'_SAVE.h5')

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

def get_person_ids(cfg):
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')
    return p_df.loc[:, "PERSON_ID"].values.tolist()