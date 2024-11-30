import xgboost as xgb
import numpy as np
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import analysis.read_data as rd
import shap

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


threshold = 0.95

DATASET_PATH = '../dataset.csv'
model_name = 'model_custom_scaler_with_f1-score.keras'

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
y_data = np.array(data[:, 1] - 1)
for i in range(len(y_data)):
    if y_data[i] == 1:
        for _ in range(5):
            y_data = np.append(y_data, 1)
            x_data = np.append(x_data, x_data[i:i+1, :], axis=0)


def normalizer(data):
    return np.array(data / (np.max(data, axis=0)), dtype=float)


y_data = np.array(y_data, dtype=float)

normal_x_data = normalizer(x_data)

x_train, x_test, y_train, y_test = train_test_split(normal_x_data, y_data, test_size=0.2)

y_train = np.array(y_train, dtype=float).reshape((-1, 1))
y_test = np.array(y_test, dtype=float).reshape((-1, 1))

x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)

# Создание объекта DMatrix для XGBoost
dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test, label=y_test)
print(dtest)

# Задание параметров модели
params = {
    'objective': 'binary:logistic',  # Задача бинарной классификации
    'eval_metric': 'logloss',        # Метрика для оценки
    'eta': 0.5,                      # Скорость обучения
    'max_depth': 8                   # Максимальная глубина дерева
}

# Обучение модели
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)  # num_boost_round - количество деревьев

# Предсказание на тестовой выборке
y_pred = model.predict(dtest)
y_pred_binary = [1 if p > threshold else 0 for p in y_pred]

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Точность: {accuracy}")

f1_score = keras.api.metrics.F1Score(threshold=threshold)
f1_score.update_state(y_test, np.array(y_pred).reshape((-1, 1)))
print('f1-score: ' + str(*f1_score.result().numpy()))


explainer = shap.Explainer(model)
shap_values = explainer(x_test)

# Построение графика
shap.summary_plot(shap_values, x_test, feature_names=necessary_columns_name)
shap.summary_plot(shap_values, x_test, feature_names=necessary_columns_name, plot_type="bar")






# model.save_model('xgboost_model_test.model')

# y_sort = sorted(y_pred)
# plt.scatter([x for x in range(len(y_sort))], y_sort)
# plt.show()

# loaded_model = xgb.Booster()
# loaded_model.load_model('xgboost_model.model')

## Подбор трешхолда
# scores = []
# for i in range(10, 100):
#     y_pred_binary = [1 if p > i / 100 else 0 for p in y_pred] # Преобразование вероятностей в бинарные классы
#     # y_sort = sorted(y_pred)
#     # plt.scatter([x for x in range(len(y_sort))], y_sort)
#     # plt.legend(str(i) + ' treshhold')
#     # plt.show()
#
#     # Оценка точности
#     accuracy = accuracy_score(y_test, y_pred_binary)
#     scores.append([i, accuracy])
#     # print(f"Точность: {accuracy}")

# scores = sorted(scores, key=lambda x: x[1], reverse=True)
# print(scores[:5])

