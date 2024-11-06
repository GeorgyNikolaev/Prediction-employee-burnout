import numpy as np
import pandas as pd
from pathlib import Path


alf = {'Yes': 0,
       'No': 1,
       'Travel_Rarely': 0,
       'Travel_Frequently': 1,
       'Non-Travel': 2,
       'Research & Development': 0,
       'Sales': 1,
       'Human Resources': 2,
       'Life Sciences': 0,
       'Medical': 1,
       'Marketing': 2,
       'Technical Degree': 3,
       'Other': 4,
       'Male': 0,
       'Female': 1,
       'Sales Executive': 0,
       'Research Scientist': 1,
       'Laboratory Technician': 2,
       'Manufacturing Director': 3,
       'Healthcare Representative': 4,
       'Manager': 5,
       'Sales Representative': 6,
       'Research Director': 7,
       'Married': 0,
       'Single': 1,
       'Divorced': 2
       }


def convert_data(data) -> np.array:
    keys = list(alf.keys())
    for i in range(len(alf)):
        x = keys[i]
        data.replace(x, alf[x], inplace=True)
    return np.array(data)


def read_data(path: str):
    pd.set_option('future.no_silent_downcasting', True)

    # Читаем датасет.
    dataset = pd.read_csv(path)

    # Читаем только нужные нам столбцы.
    BASE_DIR = Path(__file__).resolve().parent.parent
    necessary_columns_name = [x[:-1] for x in open(f'{BASE_DIR}/analysis/necessary_columns.txt', 'r').readlines()]
    data = dataset[necessary_columns_name].copy()

    # Заменяем строки на числа.
    data = convert_data(data) + 1

    return np.array(data), necessary_columns_name
