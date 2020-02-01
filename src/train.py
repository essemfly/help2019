import os
import pandas as pd
from datetime import date
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
#from tensorflow.python.client import device_lib 
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import f1_score

from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .preprocessing import exupperlowers
from .featureextraction import extract_df

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))

def main(env):
    print("Train function runs")
    train(env)


def train(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    print(device_lib.list_local_devices())
    m_df = pd.read_csv(cfg.TRAIN_DIR + measurement_csv, encoding='CP949')
    o_df = pd.read_csv(cfg.TRAIN_DIR + outcome_cohort_csv, encoding='CP949')
    feature_X = extract_df(m_df, o_df, column_list=MEASUREMENT_SOURCE_VALUE_USES)
    feature_X.to_csv(cfg.LOG_DIR +"/featuremap.csv", mode='w')
    sc = StandardScaler()
    X = sc.fit_transform(feature_X.values)
    Y = o_df['LABEL'].values

    classifier = Sequential()
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = feature_X.shape[1]))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss=[focal_loss(gamma=2.,alpha=.25)], metrics = ['accuracy'])
    classifier.fit(X, Y, batch_size = 20, epochs = 100)
    
    model_json = classifier.to_json()
    with open(cfg.LOG_DIR + '/m_' + str(ID) +'.json', "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights(cfg.LOG_DIR + '/m_' + str(ID) +'.h5')
    print("Saved model to disk")

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

def get_person_ids(cfg):
    p_df = pd.read_csv(cfg.TRAIN_DIR + person_csv, encoding='CP949')
    return p_df.loc[:, "PERSON_ID"].values.tolist()