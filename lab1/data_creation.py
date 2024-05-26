#Импортируем необходимые библиотеки
import os
import numpy as np
import pandas as pd

# Функция для генерации синтетических данных
def generate_data(size, noise_level=0.1, with_anomalies=False):
    x = np.linspace(0, 10, size)  # Генерация линейно распределенных значений от 0 до 10
    y = np.sin(x) + noise_level * np.random.randn(size)  # Генерация синусоидальных значений с добавлением шума

    # Добавление аномалий в данные, если указан параметр with_anomalies
    if with_anomalies:
        num_anomalies = int(0.1 * size)  # Количество аномалий составляет 10% от общего размера данных
        anomaly_indices = np.random.choice(size, num_anomalies, replace=False)  # Случайный выбор индексов для аномалий
        y[anomaly_indices] += np.random.normal(10, 2, num_anomalies)  # Добавление аномалий к значениям y

    return pd.DataFrame({'x': x, 'y': y})  # Возвращение данных в виде DataFrame

# Функция для сохранения данных в файлы
def save_data():
    os.makedirs('train', exist_ok=True)  # Создание директории "train", если она еще не существует
    os.makedirs('test', exist_ok=True)  # Создание директории "test", если она еще не существует

    # Цикл для генерации и сохранения нескольких наборов данных для обучения и тестирования
    for i in range(1, 4):
        # Генерация и сохранение набора данных для обучения
        data = generate_data(100 + i*10, noise_level=0.1)
        data.to_csv(f'train/data_{i}.csv', index=False)  # Сохранение данных в CSV-файл без индексов

        # Генерация и сохранение набора данных для тестирования с аномалиями
        data = generate_data(50 + i*5, noise_level=0.2, with_anomalies=True)
        data.to_csv(f'test/data_{i}.csv', index=False)  # Сохранение данных в CSV-файл без индексов

#Если скрипт был вызван для исполнения (не был импортирован как модуль), запускаем функцию
if __name__ == "__main__":
    save_data()

