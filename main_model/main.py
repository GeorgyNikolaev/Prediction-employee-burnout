import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import recommend_model.recommend as recommend
from pathlib import Path
import numpy as np
import analysis.read_data as rd
import keras


def normalizer(data):
    return np.array(data / (np.max(data, axis=0)), dtype=float)


BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATASET_PATH = BASE_DIR + '/dataset.csv'
BURNOUT_MODEL_PATH = BASE_DIR + '/burnout_model/model_custom_scaler.keras'
burnout_treshhold = 0.5

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
normal_x_data = normalizer(x_data)
y_data = np.array(data[:, 1] - 1)

model = keras.api.models.load_model(BURNOUT_MODEL_PATH)

employ = np.array(normal_x_data[np.random.randint(0, len(x_data))])

burnout = model(employ.reshape(1, -1)).numpy().reshape(-1)[0]
is_burnout = True if burnout > burnout_treshhold else False

recommendations_text = recommend.recommendation(employ)

if is_burnout:
    print('Ваш сотрудник выгорел')
else:
    print('Ваш сотрудник не выгорел')

print('Вот несколько рекомендаций по улучшению состояния сотрудника')
print(recommendations_text)

