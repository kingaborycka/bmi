import tensorflow as tf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# cols=['cholesterol', 'glucose', 'age', 'gender', 'height', 'weight', 'bmi']
cols=['age', 'gender', 'height', 'weight', 'bmi']

# wczytanie danych z pliku
dataset = pd.read_csv('./diabetes.csv', usecols = cols)

# przygotowanie danych
dataset['bmi'] = dataset['bmi'].str.replace(',','.')
dataset['gender'] = dataset['gender'].str.replace('female','0')
dataset['gender'] = dataset['gender'].str.replace('male','1')
 
for i in cols: dataset[i] = dataset[i].astype(float)

## podział danych na uczące i testowe
train_dataset = dataset.sample(frac=0.8, random_state=0) # 312 wierszy
test_dataset = dataset.drop(train_dataset.index)         # 78 wierszy

## sprawdzenie danych uczących
print(test_dataset.head())

## oczekiwane wartości
train_labels = train_dataset.pop('bmi') 
test_labels = test_dataset.pop('bmi')   

print('\nTRAIN DATASET')
print(train_dataset.head())
print('\nTRAIN LABELS')
print(train_labels.head())
print('\nTEST DATASET')
print(test_dataset.head())
print('\nTEST LABELS')
print(test_labels.head())

## normalizacja danych
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_dataset))

## definiowanie warstw modelu
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
    epochs = 200,
    # na 20% danych treningowych zostaną obliczone wyniki walidacji
    validation_split = 0.2)

end_learning_time = time.time()

## wykres funkcji błędu
def plot_loss(history):
  plt.plot(history.history['loss'], "r--")
  plt.plot(history.history['val_loss'], "g--")
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Loss [BMI]')
  plt.legend(['train', 'test'], loc='best')
  plt.grid(True)
  plt.show()

print(model.summary())

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
