import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import recommend_model.recommend as recommend
from pathlib import Path
import numpy as np
import analysis.read_data as rd
import keras


BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATASET_PATH = BASE_DIR + '/dataset.csv'
BURNOUT_MODEL_PATH = BASE_DIR + '/burnout_model/model_custom_scaler.keras'

burnout_treshhold = 0.5

def normalizer_employ(data):
    max_values = np.load(file=BASE_DIR + '/main_model/max_values_data.npy', allow_pickle=True)
    max_values = np.concatenate((max_values[:1], max_values[2:]))
    return np.array(data / max_values, dtype=float)


def SLON(employ, isTest: bool=False, y_true: int=0):
    results = ''
    model = keras.api.models.load_model(BURNOUT_MODEL_PATH)

    x_employ = normalizer_employ(employ)
    burnout = model(x_employ.reshape(1, -1)).numpy().reshape(-1)[0]

    recommendations_text = recommend.recommendation(employ)
    # print()
    if burnout > burnout_treshhold:
        results += 'Ваш сотрудник выгорел\n\n'
        # print('Ваш сотрудник выгорел')
    else:
        results += 'Ваш сотрудник не выгорел\n\n'
        # print('Ваш сотрудник не выгорел')

    # print('Вот несколько рекомендаций по улучшению состояния сотрудника:')
    results += 'Вот несколько рекомендаций по улучшению состояния сотрудника:\n'
    for line in recommendations_text[:-1].split('\n'):
        # print('-' + line)
        results += '-' + line + '\n'
    
    return results


data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
normal_x_data = normalizer_employ(x_data)
y_data = np.array(data[:, 1] - 1)

employ = np.array(x_data[np.random.randint(0, len(x_data))])

# SLON(employ=employ)



