from pathlib import Path
import numpy as np
import analysis.read_data as rd
import json

BASE_DIR = str(Path(__file__).resolve().parent.parent)

with open(BASE_DIR + '\\recommend_model\\recommendations_text.json', 'r', encoding='utf-8') as file:
    recommendations_text = json.load(file)
# print(recommendations_text)

def load_recommend_standard_values():
    recommend_columns_name = open(BASE_DIR + '/recommend_model/recommend_columns', 'r').readlines()
    recommend_standard_values = {}
    for i in range(len(recommend_columns_name)):
        name, left, middle, right = map(str, recommend_columns_name[i].split(' '))
        left = int(left)
        middle = int(middle)
        right = int(right)
        standard_values = dict()
        standard_values['critical'] = [left, middle]
        standard_values['non_critical'] = [middle, right]
        recommend_standard_values[name] = standard_values
    return recommend_standard_values

recommend_standard_values = load_recommend_standard_values()
# print(recommend_standard_values)

def recommendation(employ: np.array):
    recommend = ''
    for x, y in zip(employ, necessary_columns_name):
        rec = get_recommendation(y, x)
        if rec is not None and rec != '':
            recommend += rec + '\n'
    return recommend


def get_recommendation(feature: str, value: int) -> str:
    if feature in recommend_standard_values:
        # print(feature + ': ' + str(value))
        value_critical = recommend_standard_values[feature]['critical']
        value_non_critical = recommend_standard_values[feature]['non_critical']
        # print(value_critical, value_non_critical)

        if isBetween(value=value, arr=value_critical):
            return recommendations_text[feature]['critical']

        elif isBetween(value=value, arr=value_non_critical):
            return recommendations_text[feature]['non_critical']

        else:
            return ''
    return ''


def isBetween(value: int, arr: list) -> bool:
    if arr[0] <= value < arr[1] or arr[1] < value <= arr[0]:
        return True
    else:
        return False


DATASET_PATH = BASE_DIR + '/dataset.csv'

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

# x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)

# employ = x_data[np.random.randint(0, len(x_data))]
# recommendation_text = recommendation(employ)
# print(employ)
# print(recommendation_text)




