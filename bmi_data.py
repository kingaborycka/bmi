import tensorflow as tf
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers


cols = ['Sex','Age','Height(Inches)','Weight(Pounds)','BMI']
# cols = ['Age','Height(Inches)','Weight(Pounds)','BMI']

# wczytanie danych z pliku
dataset = pd.read_csv('./bmi_data.csv', usecols=cols)

# usunięcie wierszy z pustymi wartościami
dataset = dataset.dropna()

# przygotowanie danych
dataset['Sex'] = dataset['Sex'].str.replace('Female','0')
dataset['Sex'] = dataset['Sex'].str.replace('Male','1')

for i in cols: dataset[i] = dataset[i].astype(float)

# podział danych na uczące i testowe
train_dataset = dataset.sample(frac=0.8, random_state=0) # 19960 wierszy
test_dataset = dataset.drop(train_dataset.index)         # 4990  wierszy

## sprawdzenie danych uczących
print(test_dataset.head())

## oczekiwane wartości
train_labels = train_dataset.pop('BMI') 
test_labels = test_dataset.pop('BMI')   

print('\nTRAIN DATASET')
print(train_dataset.head())
print('\nTRAIN LABELS')
print(train_labels.head())
print('\nTEST DATASET')
print(test_dataset.head())
print('\nTEST LABELS')
print(test_labels.head())

## normalizacja danych
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

## definiowanie struktury modelu
model = tf.keras.Sequential([
    normalizer,
    # warstwa ukryta
    layers.Dense(units=16, activation='sigmoid', input_dim=4),

    # warstwa wyjściowa
    layers.Dense(units=1)
])

## konfiguracja modelu
## wybór optymalizatora, funkcji błędu oraz miernika jakości
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate = 0.1),
    loss='mean_squared_error')

start_learning_time = time.time()

## wywołanie procesu uczenia
history = model.fit(
    train_dataset,
    train_labels,
    batch_size=32,
    epochs = 3,
    # na 20% danych treningowych zostaną obliczone wyniki walidacji
    validation_split = 0.2)

end_learning_time = time.time()

## wykres funkcji błędu
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.2])
  plt.xlabel('Epoch')
  plt.ylabel('Error [BMI]')
  plt.legend()
  plt.grid(True)
  plt.show()

print("Czas uczenia: ", end_learning_time-start_learning_time)
plot_loss(history)

test_results = model.evaluate(
    test_dataset, test_labels)

print("Test results: \n'loss': ",test_results)

## Wykres prognoz modelu
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [BMI]')
plt.ylabel('Predictions [BMI]')
lims = [0, 40]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

