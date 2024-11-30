import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import recommend_model.recommend as recommend
from pathlib import Path
import numpy as np
import analysis.read_data as rd
import keras
import xgboost as xgb


BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATASET_PATH = BASE_DIR + '/dataset.csv'
# BURNOUT_MODEL_PATH = BASE_DIR + '/burnout_model/model_custom_scaler.keras'
BURNOUT_MODEL_PATH = BASE_DIR + '/burnout_model/xgboost_model.model'

burnout_low_treshhold = 0.25
burnout_up_treshhold = 0.95

def normalizer_employ(data):
    max_values = np.load(file=BASE_DIR + '/main_model/max_values_data.npy', allow_pickle=True)
    return np.array(data / max_values, dtype=float)


def SLON(employ, isTest: bool=False, y_true: int=0):
    results = ''
    # model = keras.api.models.load_model(BURNOUT_MODEL_PATH)
    model = xgb.Booster()
    model.load_model(BURNOUT_MODEL_PATH)

    x_employ = normalizer_employ(employ).reshape((1, -1))
    # burnout = model(x_employ.reshape(1, -1)).numpy().reshape(-1)[0]
    x_employ = xgb.DMatrix(data=x_employ)
    burnout = model.predict(x_employ)[0]

    recommendations_text = recommend.recommendation(employ)

    if burnout > burnout_up_treshhold:
        results += 'У вас высокая вероятность выгореть.\n\n'
    elif burnout_low_treshhold < burnout < burnout_up_treshhold:
        results += 'Ваше состояние можно оценить как нормальное.\n\n'
    else:
        results += 'Ура, у вас низкая вероятность выгореть.\n\n'

    if recommendations_text != "":
        results += ('Дальше, вне зависимости от вашего состояния мы предложим вам ряд мер по улучшению '
                    'вашего состояния:\n\n')
        for line in recommendations_text[:-1].split('\n'):
            if line != "":
                results += '-' + line + '\n\n'
            else:
                results += ""
    else:
        results += "Все отлично!\nОтклонений не выявлено."
    return results


# data, necessary_columns_name = rd.read_data(DATASET_PATH)
# necessary_columns_name.remove('Attrition')
#
# x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
# normal_x_data = normalizer_employ(x_data)
# y_data = np.array(data[:, 1] - 1)
#
# employ = np.array(x_data[np.random.randint(0, len(x_data))])
#
# print(SLON(employ=employ))



