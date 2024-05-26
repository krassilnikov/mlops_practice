#Импортируем необходимые библиотеки
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Создаем экземпляр модели линейной регрессии
    model = LinearRegression()

    # Формируем список файлов в папке 'train', имена которых начинаются с 'scaled_'
    files = [f for f in os.listdir('train') if f.startswith('scaled_')]

    # Создаем пустой список для хранения всех загруженных данных
    all_data = []

    # Цикл по каждому файлу в списке files
    for file in files:
        # Читаем CSV файл и добавляем его в список all_data
        data = pd.read_csv(os.path.join('train', file))
        all_data.append(data)

    # Объединяем все данные в один DataFrame
    train_data = pd.concat(all_data)

    # Извлекаем колонки 'x' и 'y' для обучения модели
    X = train_data['x'].values.reshape(-1, 1)
    y = train_data['y'].values

    # Обучаем модель на предоставленных данных
    model.fit(X, y)

    # Создаем директорию 'models' при необходимости
    os.makedirs('models', exist_ok=True)

    # Сохраняем обученную модель в файл 'model.joblib'
    joblib.dump(model, 'models/model.joblib')

#Если скрипт был вызван для исполнения (не был импортирован как модуль), запускаем функцию обучения модели
if __name__ == "__main__":
    train_model()