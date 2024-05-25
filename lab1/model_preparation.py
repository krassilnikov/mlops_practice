from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

train_data = np.loadtxt("processed_data/train_data_scaled.csv", delimiter=",")

# Пример обучения модели
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "trained_model.pkl")