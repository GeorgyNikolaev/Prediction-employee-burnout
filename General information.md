# Тут мы собрали всю информацию про выгорание сотрудников

---

## Общая информация

### Примеры
1. **Fade**<br>
   Это сервис для оценки выгорания сотрудников

### Статистика
 - Около 50% выгорали
 - Выгоревший сотрудник приносит большие убытки

### Причины выгорания
 1. Нервная обстановка на работе
 2. Неадекватная оценка работы сотрудника
 3. Несоразмерная финансовая мотивация

### Ссылки
- [Are Your Employees Burning Out?](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out?select=test.csv)<br>
Тут есть столбик с burnout так что это то, что нам надо, но мало характеристик о сотруднике (9 учитывая индентификатор сотрудника). <br>И еще, видимо это данные с хакатона.
Вот несколько решений: [вот](https://github.com/hemnaresh/employee-burnout-prediction) и [вот](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out?select=train.csv), даже [видос](https://www.youtube.com/watch?v=D3yyRMI_RTA) есть.
- [Employee dataset](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset)<br>
Тут просто данные о сотрудниках (город, образование, пол, возраст). Целевая колонка `Leave or not`.
- [Employee Attrition](https://www.kaggle.com/datasets/HRAnalyticRepository/employee-attrition-data)<br>
Фейковые данные о сотрудниках для тренировки предсказания увольнения сотрудника.
- [Employee Turnover](https://www.kaggle.com/datasets/davinwijaya/employee-turnover/data)<br>
Реальные данные о сотрудниках для анализа `Выживаемости`. Тут есть профессия, область работы, опыт, способ передвижения сотрудника, тревожность,
много всяких оценок и т.д.
- [HR Analytics](https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction)<br>
Тут есть колока с attrition, расстояние до дома, вовлеченность сотрудника, удовлетворен ли сотрудник и многое другого.

---

## Идеи
#### Что учитывать в данных о сотруднике?
 - Интерес к работе
 - Зарплата на рынке и зарплата сотрудника (можно как коэффициент)
