import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Функция для предварительной обработки данных и их сохранения
def preprocess_and_save(directory):
    # Получить список файлов с расширением .csv в указанной директории
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    scaler = StandardScaler()

    for file in files:
        # Чтение данных из .csv файла
        data = pd.read_csv(os.path.join(directory, file))
        # Масштабирование данных с использованием StandardScaler
        scaled_data = scaler.fit_transform(data)
        # Преобразование масштабированных данных обратно в DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
        # Сохранение масштабированных данных в новый .csv файл
        scaled_df.to_csv(os.path.join(directory, f'scaled_{file}'), index=False)

    # Возвращение объекта StandardScaler для дальнейшего использования
    return scaler

#Если скрипт был вызван для исполнения (не был импортирован как модуль), запускаем функцию
if __name__ == "__main__":
    # Создание директории 'scalers', если она еще не существует
    #os.makedirs('scalers', exist_ok=True)
    # Предварительная обработка файлов в директории 'train' и сохранение объекта scaler
    scaler = preprocess_and_save('train')
    preprocess_and_save('test')
    # Сохранение объекта scaler в файл для последующего использования
    #joblib.dump(scaler, 'scalers/scaler.joblib')

