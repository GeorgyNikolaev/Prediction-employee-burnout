import os

import numpy

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

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')
recommend_columns_name = open('recommend_columns', 'r').readlines()

def load_recommend_standard_values():
    recommend_standard_values = {}
    for i in range(len(recommend_columns_name)):
        name, left, right = map(str, recommend_columns_name[i].split(' '))
        left = int(left)
        right = int(right)
        recommend_standard_values[name] = [left, right]

print(np.load('recommend_standard_values.npy', allow_pickle=True))


# x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
#
# for i, name in enumerate(necessary_columns_name):
#     if name in recommend_columns_name:
#         print(name)
#         print('std: ' + str(round(np.std(x_data[:, i]), 2)))
#         print('mean: ' + str(round(np.mean(x_data[:, i]), 2)))
#         print('max: ' + str(round(np.max(x_data[:, i]), 2)))
#         print('min: ' + str(round(np.min(x_data[:, i]), 2)))
#         print()


