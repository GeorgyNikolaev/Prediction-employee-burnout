from pathlib import Path
import numpy as np
import analysis.read_data as rd


recommendations_treshold = {
    'OverTime': 1,
    'JobSatisfaction': 3,
    'WorkLifeBalance': 3,
    'DistanceFromHome': 15,
    'JobInvolvement': 3,
    'RelationshipSatisfaction': 3,
    'YearsSinceLastPromotion': 3

}
predefined_recommendations = {
    'OverTime': "Рассмотрите возможность уменьшения количества сверхурочных часов. Попробуйте распределить задачи более равномерно или обсудите гибкий график работы с руководством.",
    'BusinessTravel': "Уменьшите количество деловых поездок, если это возможно. Рассмотрите возможность проведения встреч онлайн.",
    'DistanceFromHome': "Если возможно, рассмотрите возможность удалённой работы или гибкого графика, чтобы сократить время на дорогу до работы.",
    'JobSatisfaction': "Обсудите с руководством возможности повышения удовлетворённости работой, такие как улучшение условий труда или предоставление возможностей для профессионального роста.",
    'WorkLifeBalance': "Старайтесь поддерживать баланс между работой и личной жизнью. Рассмотрите возможность гибкого графика или удалённой работы.",
    'JobInvolvement': "Участвуйте в проектах и инициативах компании, чтобы повысить вовлечённость в работу.",
    'RelationshipSatisfaction': "Работайте над улучшением отношений с коллегами и руководством через командные мероприятия и открытое общение.",
    'YearsSinceLastPromotion': "Обсудите возможности карьерного роста и продвижения с руководством.",
    'JobLevel': "Рассмотрите возможности повышения уровня вашей должности через обучение и развитие навыков.",
    'PercentSalaryHike': "Обсудите вопросы компенсации и повышения зарплаты с руководством.",
    'YearsWithCurrManager': "Если отношения с текущим менеджером напряжённые, рассмотрите возможность смены менеджера или участия в тренингах по управлению конфликтами.",
    'JobRole': "Исследуйте возможности смены роли внутри компании для повышения удовлетворённости работой.",
    'MonthlyRate': "Обсудите вопросы вознаграждения и дополнительных бонусов с руководством.",
    'MonthlyIncome': "Рассмотрите возможности повышения дохода через дополнительные проекты или повышение квалификации."
}


def recommendation(employ: np.array):
    recommend = ''
    for x, y in zip(employ, necessary_columns_name):
        rec = get_recommendation(y, x)
        if rec is not None and rec != '':
            recommend += rec + '\n'
    return recommend

def get_recommendation(feature, value):
    # Определяет рекомендацию на основе признака и его значения
    if feature == 'OverTime':
        if value == 1:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'JobSatisfaction':
        if value < 3:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'WorkLifeBalance':
        if value < 3:
            return predefined_recommendations[feature]
        else:
            return "Продолжайте поддерживать хороший баланс между работой и личной жизнью."
    elif feature == 'DistanceFromHome':
        if value > 15:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'JobInvolvement':
        if value < 3:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'RelationshipSatisfaction':
        if value < 3:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'YearsSinceLastPromotion':
        if value > 3:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'JobLevel':
        if value < 3:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'PercentSalaryHike':
        if value < 15:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'YearsWithCurrManager':
        if value > 5:
            return predefined_recommendations[feature]
        else:
            return None
    elif feature == 'JobRole':
        return predefined_recommendations.get(feature, "")
    elif feature == 'MonthlyRate':
        return predefined_recommendations.get(feature, "")
    elif feature == 'MonthlyIncome':
        if value < 5000:
            return predefined_recommendations[feature]
        else:
            return None
    else:
        return predefined_recommendations.get(feature, "")


BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATASET_PATH = BASE_DIR + '/dataset.csv'

data, necessary_columns_name = rd.read_data(DATASET_PATH)
necessary_columns_name.remove('Attrition')

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
y_data = np.array(data[:, 1] - 1)

employ = x_data[np.random.randint(0, len(x_data))]






