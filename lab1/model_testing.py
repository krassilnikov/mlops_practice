# Импортируем необходимые библиотеки
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Основная функция для тестирования модели
def test_model():
    # Загружаем масштабировщик данных и машинную модель
    #scaler = joblib.load('scalers/scaler.joblib')
    model = joblib.load('models/model.joblib')

    # Получаем список файлов из директории "test", имена которых начинаются с "scaled_"
    files = [f for f in os.listdir('test') if f.startswith('scaled_')]
    all_rmses = []

    # Проходимся по каждому файлу в списке
    for file in files:
        # Читаем данные из файла в DataFrame
        data = pd.read_csv(os.path.join('test', file))
        # Извлекаем входные данные (предикторы) и преобразуем их в необходимую форму
        X = data['x'].values.reshape(-1, 1)
        # Извлекаем истинные значения (таргеты)
        y_true = data['y'].values
        # Предсказываем значения с помощью модели
        y_pred = model.predict(X)

        # Рассчитываем корень из среднеквадратической ошибки (RMSE)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        #print (rmse)
        # Добавляем полученный RMSE в список
        all_rmses.append(rmse)
        # Выводим RMSE для текущего файла
        print(f'RMSE for {file} : {rmse}')

    # Рассчитываем средний RMSE для всех файлов
    avg_rmse = np.mean(all_rmses)
    # Выводим средний RMSE
    print(f'Average RMSE: {avg_rmse}')

# Если скрипт был вызван для исполнения (не был импортирован как модуль), запускаем функцию
if __name__ == "__main__":
    test_model()