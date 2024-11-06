import matplotlib.pyplot as plt
import pandas as pd
import read_data
import seaborn as sns

DATASET_PATH = '../dataset.csv'
dataset = pd.read_csv(DATASET_PATH)

# Читаем только нужные нам столбцы.
necessary_columns_name = [x[:-1] for x in open('necessary_columns', 'r').readlines()]
data = dataset[necessary_columns_name]

# Заменяем строки на числа.
data_conv = read_data.convert_data(data)


# Для удобства визуализации используем pandas.
data_conv_df = pd.DataFrame(data_conv, columns=necessary_columns_name)

# Строим корреляционную матрицу.
correlation_matrix_df = data_conv_df.corr()

# Визуализация корреляционной матрицы.
sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm')
plt.show()

# Строим все возможные графики ради интереса.
for i in range(len(data_conv[0])-1):
    for j in range(len(data_conv[0])):
        plt.scatter(data_conv[:, i], data_conv[:, j], s=1)
        plt.xlabel(necessary_columns_name[i])
        plt.ylabel(necessary_columns_name[j])
        plt.show()

