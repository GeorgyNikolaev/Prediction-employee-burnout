# from pathlib import Path
# import shap
# import pandas as pd
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import joblib
# import read_data
#
# PROCESSED_DATA_PATH = 'recommend/new_dataset.csv'
# MODEL_PATH = 'first_model/first_model.keras'
# SCALER_PATH = 'recommend/scalers/standard_scaler.pkl'
# NORMALIZER_PATH = 'recommend/scalers/normalizer.pkl'
# EXPLAINER_PATH = "recommend/shap_explainer.pkl"
#
# actionable_features = [
#     'OverTime',
#     'BusinessTravel',
#     'DistanceFromHome',
#     'JobSatisfaction',
#     'WorkLifeBalance',
#     'JobInvolvement',
#     'RelationshipSatisfaction',
#     'YearsSinceLastPromotion',
#     'JobLevel',
#     'PercentSalaryHike',
#     'YearsWithCurrManager',
#     'JobRole',
#     'MonthlyRate',
#     'MonthlyIncome'
# ]
#
# non_actionable_features = [
#     'NumCompaniesWorked',
#     'TotalWorkingYears',
#     'YearsInCurrentRole',
#     'Age',
#     'Department',
#     'Gender',
#     'MaritalStatus',
#     'EducationField',
#     'HourlyRate',
#     'YearsAtCompany',
#     'DailyRate',
#     'Education'
# ]
#
# predefined_recommendations = {
#     'OverTime': "Рассмотрите возможность уменьшения количества сверхурочных часов. Попробуйте распределить задачи более равномерно или обсудите гибкий график работы с руководством.",
#     'BusinessTravel': "Уменьшите количество деловых поездок, если это возможно. Рассмотрите возможность проведения встреч онлайн.",
#     'DistanceFromHome': "Если возможно, рассмотрите возможность удалённой работы или гибкого графика, чтобы сократить время на дорогу до работы.",
#     'JobSatisfaction': "Обсудите с руководством возможности повышения удовлетворённости работой, такие как улучшение условий труда или предоставление возможностей для профессионального роста.",
#     'WorkLifeBalance': "Старайтесь поддерживать баланс между работой и личной жизнью. Рассмотрите возможность гибкого графика или удалённой работы.",
#     'JobInvolvement': "Участвуйте в проектах и инициативах компании, чтобы повысить вовлечённость в работу.",
#     'RelationshipSatisfaction': "Работайте над улучшением отношений с коллегами и руководством через командные мероприятия и открытое общение.",
#     'YearsSinceLastPromotion': "Обсудите возможности карьерного роста и продвижения с руководством.",
#     'JobLevel': "Рассмотрите возможности повышения уровня вашей должности через обучение и развитие навыков.",
#     'PercentSalaryHike': "Обсудите вопросы компенсации и повышения зарплаты с руководством.",
#     'YearsWithCurrManager': "Если отношения с текущим менеджером напряжённые, рассмотрите возможность смены менеджера или участия в тренингах по управлению конфликтами.",
#     'JobRole': "Исследуйте возможности смены роли внутри компании для повышения удовлетворённости работой.",
#     'MonthlyRate': "Обсудите вопросы вознаграждения и дополнительных бонусов с руководством.",
#     'MonthlyIncome': "Рассмотрите возможности повышения дохода через дополнительные проекты или повышение квалификации."
# }
#
# def get_recommendations(employee: pd.DataFrame, actionable_features=actionable_features, top_n=3) -> pd.DataFrame:
#     # Препроцессинг данных сотрудника
#     employee_processed = read_data.convert_data(employee)
#
#     # Загрузка модели и предобработчиков
#     model = keras.models.load_model(MODEL_PATH)
#     standard_scaler = joblib.load(SCALER_PATH)
#     normalizer = joblib.load(NORMALIZER_PATH)
#     explainer = joblib.load(EXPLAINER_PATH)
#
#     # Вычисление SHAP-значений и формирование рекомендаций
#     employee_scaled = standard_scaler.transform(employee_processed)
#     employee_normalized = normalizer.transform(employee_scaled)
#     feature_names = employee.columns.tolist()
#     shap_values = explainer(employee_normalized)
#     shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
#
#     for idx in range(len(employee)):
#         employee_shap = shap_values_df.iloc[idx]
#         employee_values = employee.iloc[idx]
#         actionable_shap = pd.Series(index=actionable_features)
#         for feature in actionable_features:
#             columns = [col for col in feature_names if col.startswith(feature)]
#             shap_sum = employee_shap[columns].sum()
#             actionable_shap[feature] = shap_sum
#         sorted_features = actionable_shap.abs().sort_values(ascending=False).index.tolist()
#         rec_list = []
#         for feature in sorted_features:
#             value = employee_values[feature]
#             recommendation = get_recommendation(feature, value)
#             if recommendation:
#                 rec_list.append(recommendation)
#                 if len(rec_list) == top_n:
#                     break
#         rec_text = ' '.join(rec_list)
#         employee.at[employee.index[idx], 'recommendation'] = rec_text
#     return employee
#
# def get_recommendation(feature, value):
#     # Определяет рекомендацию на основе признака и его значения
#     if feature == 'OverTime':
#         if value == 1:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'JobSatisfaction':
#         if value < 3:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'WorkLifeBalance':
#         if value < 3:
#             return predefined_recommendations[feature]
#         else:
#             return "Продолжайте поддерживать хороший баланс между работой и личной жизнью."
#     elif feature == 'DistanceFromHome':
#         if value > 15:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'JobInvolvement':
#         if value < 3:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'RelationshipSatisfaction':
#         if value < 3:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'YearsSinceLastPromotion':
#         if value > 3:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'JobLevel':
#         if value < 3:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'PercentSalaryHike':
#         if value < 15:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'YearsWithCurrManager':
#         if value > 5:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     elif feature == 'JobRole':
#         return predefined_recommendations.get(feature, "")
#     elif feature == 'MonthlyRate':
#         return predefined_recommendations.get(feature, "")
#     elif feature == 'MonthlyIncome':
#         if value < 5000:
#             return predefined_recommendations[feature]
#         else:
#             return None
#     else:
#         return predefined_recommendations.get(feature, "")
#
#
# # Загрузка и предобработка данных
# data = pd.read_csv("../dataset.csv").drop("Attrition", axis=1)
# data = data[data["OverTime"] != "Yes"]
# print(data['OverTime'].value_counts()['No'])
# print(len(data))
# # necessary_columns_path = 'C:\\Users\\MSI\\Prediction-employee-burnout\\analysis\\necessary_columns'
# BASE_DIR = str(Path(__file__).resolve().parent.parent)
# necessary_columns_path = BASE_DIR + '/analysis/necessary_columns.txt'
# with open(necessary_columns_path, 'r') as file:
#     necessary_columns_name = [line.strip() for line in file.readlines()]
# person = data.iloc[:100]
# necessary_columns_name.remove("Attrition")
# person = person[necessary_columns_name]
# # Получение рекомендаций
# result = get_recommendations(person, top_n=3)
# # Вывод рекомендаций
# for rec in result['recommendation']:
#     print("*"*30)
#     print(rec)