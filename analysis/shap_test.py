import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
import pandas as pd
import analysis.read_data as rd
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import shap
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
model_name = 'model_custom_scaler.keras'
path_to_model = f'{BASE_DIR}/burnout_model/' + model_name

DATASET_PATH = '../dataset.csv'

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name = necessary_columns_name[:1] + necessary_columns_name[2:]

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
y_data = np.array(data[:, 1] - 1)


def normalizer(data):
    return np.array(data / (np.max(data, axis=0)), dtype=float)


y_data = np.array(y_data, dtype=float)

# standard = StandardScaler()
# standard_x_data = standard.fit_transform(x_data)
# normal = Normalizer()
# normal_x_data = normal.fit_transform(standard_x_data)
normal_x_data = normalizer(x_data)

X_train, X_test, y_train, y_test = train_test_split(normal_x_data, y_data, test_size=0.15)


# Загрузите вашу модель Keras
model = keras.models.load_model(path_to_model)

explainer = shap.DeepExplainer(model, X_train)
# shap_values = np.array(explainer(normal_x_data[:500, :]).values).reshape((-1, 23))
shap_values = np.array(explainer(X_test).values).reshape((-1, 23))
# shap.summary_plot(shap_values, X_test, feature_names=necessary_columns_name)
shap.summary_plot(shap_values, X_test, feature_names=necessary_columns_name, plot_type="bar")
# for i, name in enumerate(necessary_columns_name):
#     # if name == 'DailyRate':
#     x = x_data[:500, i]
#     y = shap_values[:, i]
#     plt.scatter(x, y)
#     plt.xlabel(name)
#     plt.ylabel('shape value')
#     plt.show(block=True)

