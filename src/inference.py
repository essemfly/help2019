import os
import pandas as pd
from datetime import date
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.python.client import device_lib 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from .config import LocalConfig, ProdConfig
from .constants import MEASUREMENT_SOURCE_VALUE_USES, measurement_csv, outcome_cohort_csv, person_csv
from .preprocessing import exupperlowers
from .featureextraction import extract_df

ID = os.environ.get('ID', date.today().strftime("%Y%m%d"))

def inference(env):
    cfg = LocalConfig if env == 'localhost' else ProdConfig
    
    WEIGHT_FILE = cfg.LOG_DIR + '/model_save_' + str(ID) +'/'

    json_file = open(cfg.LOG_DIR + '/model_save_' + str(ID) +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights(cfg.LOG_DIR + '/model_save_' + str(ID) +'.h5')
    print("Loaded model from disk")
    classifier.compile(optimizer = 'adam', loss=[focal_loss(gamma=2.,alpha=.25)], metrics = ['accuracy'])

    m_df = pd.read_csv(cfg.TEST_DIR + measurement_csv, encoding='CP949')
    m_df = exupperlowers(m_df)  ## preprocessing by excluding predefined outliers - 200127 by SYS
    o_df = pd.read_csv(cfg.TEST_DIR + outcome_cohort_csv, encoding='CP949')
    feature_X = extract_df(m_df, o_df, column_list=MEASUREMENT_SOURCE_VALUE_USES)
    sc = StandardScaler()
    X_test = sc.fit_transform(feature_X.values)
    
    probs = classifier.predict(X_test)

    o_df["LABEL_PROBABILITY"] = probs
    o_df.loc[o_df["LABEL_PROBABILITY"] > 0.5, "LABEL"] = 1
    o_df.loc[o_df["LABEL_PROBABILITY"] <= 0.5, "LABEL"] = 0

    o_df.to_csv(cfg.OUTPUT_DIR + "/output.csv", 
                columns = ['LABEL_PROBABILITY','LABEL'],
                index = False)

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed