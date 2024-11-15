import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.api import layers
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import analysis.read_data as rd


learning_rate = 0.0001
batch_size = 1
epochs = 100
treshold = 0.5

DATASET_PATH = '../dataset.csv'
model_name = 'model_custom_scaler.keras'

data, necessary_columns_name = rd.read_data(DATASET_PATH)

x_data = np.concatenate((data[:, :1], data[:, 2:]), axis=1)
y_data = np.array(data[:, 1] - 1)
for i in range(len(y_data)):
    if y_data[i] == 1:
        for _ in range(5):
            y_data = np.append(y_data, 1)
            x_data = np.append(x_data, x_data[i:i+1, :], axis=0)

print(list(y_data).count(0))
print(list(y_data).count(1))


def normalizer(data):
    return np.array(data / (np.max(data, axis=0)), dtype=float)


y_data = np.array(y_data, dtype=float)

# standard = StandardScaler()
# standard_x_data = standard.fit_transform(x_data)
# normal = Normalizer()
# normal_x_data = normal.fit_transform(standard_x_data)
normal_x_data = normalizer(x_data)

x_train, x_test, y_train, y_test = train_test_split(normal_x_data, y_data, test_size=0.2)
y_train = np.array(y_train, dtype=float)
y_test = np.array(y_test, dtype=float)
x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)

model = keras.api.Sequential()
model.add(layers.Dense(1024, activation='relu',))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu',))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.api.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.api.losses.binary_crossentropy,
              metrics=['accuracy'])

# Определяем коллбэки
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode="max"
)
checkpoint = ModelCheckpoint(
    filepath=model_name,
    monitor='val_accuracy',
    save_best_only=True,
    mode="max"
)

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.07,
          callbacks=[checkpoint, early_stopping])

model = keras.api.models.load_model(model_name)
predict = model.predict(x_test).reshape((-1))
y_predict = [1 if x > treshold else 0 for x in predict]

accuracy = accuracy_score(y_predict, y_test)
print(accuracy)

plt.scatter([x for x in range(len(y_test))], y_predict, c='r')
plt.scatter([x for x in range(len(y_test))], y_test, c='b', s=1)
plt.show()

plt.scatter([x for x in range(len(y_test))], sorted(predict))
plt.show()


