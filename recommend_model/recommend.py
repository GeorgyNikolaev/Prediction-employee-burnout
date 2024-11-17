from pathlib import Path
import numpy as np
import analysis.read_data as rd

BASE_DIR = str(Path(__file__).resolve().parent.parent)

predefined_recommendations = {
    'OverTime': "Рассмотрите возможность уменьшения количества сверхурочных часов. Попробуйте распределить задачи более равномерно или обсудите гибкий график работы с руководством.",
    'BusinessTravel': "Уменьшите количество деловых поездок, если это возможно. Рассмотрите возможность проведения встреч онлайн.",
    'DistanceFromHome': "Если возможно, рассмотрите возможность удалённой работы или гибкого графика, чтобы сократить время на дорогу до работы.",
    'JobSatisfaction': "Обсудите с руководством возможности повышения удовлетворённости работой, такие как улучшение условий труда или предоставление возможностей для профессионального роста.",
    'WorkLifeBalance': "Старайтесь поддерживать баланс между работой и личной жизнью. Рассмотрите возможность гибкого графика или удалённой работы.",
    'JobInvolvement': "Участвуйте в проектах и инициативах компании, чтобы повысить вовлечённость в работу.",
    'RelationshipSatisfaction': "Работайте над улучшением отношений с коллегами и руководством через командные мероприятия и открытое общение.",
    'YearsSinceLastPromotion': "Обсудите возможности карьерного роста и продвижения с руководством.",
    'MonthlyIncome': "Рассмотрите возможности повышения дохода через дополнительные проекты или повышение квалификации."
}

def load_recommend_standard_values():
    recommend_columns_name = open(BASE_DIR + '/recommend_model/recommend_columns', 'r').readlines()
    recommend_standard_values = {}
    for i in range(len(recommend_columns_name)):
        name, left, right = map(str, recommend_columns_name[i].split(' '))
        left = int(left)
        right = int(right)
        recommend_standard_values[name] = [left, right]
    return recommend_standard_values

recommend_standard_values = load_recommend_standard_values()

def recommendation(employ: np.array):
    recommend = ''
    for x, y in zip(employ, necessary_columns_name):
        rec = get_recommendation(y, x)
        if rec is not None and rec != '':
            recommend += rec + '\n'
    return recommend


def get_recommendation(feature: str, value: int) -> str:
    if feature in recommend_standard_values:
        print(feature + ': ' + str(value))
        standard_value = recommend_standard_values[feature]
        if not standard_value[0] < value < standard_value[1]:
            return predefined_recommendations[feature]
        else:
            return ''
    return ''



DATASET_PATH = BASE_DIR + '/dataset.csv'

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

# x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)

# employ = x_data[np.random.randint(0, len(x_data))]
# recommendation_text = recommendation(employ)
# print(employ)
# print(recommendation_text)




