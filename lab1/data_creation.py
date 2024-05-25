import os
import numpy as np
import pandas as pd

def generate_data(size, noise_level=0.1, with_anomalies=False):
    x = np.linspace(0, 10, size)
    y = np.sin(x) + noise_level * np.random.randn(size)
    if with_anomalies:
        num_anomalies = int(0.1 * size)
        anomaly_indices = np.random.choice(size, num_anomalies, replace=False)
        y[anomaly_indices] += np.random.normal(10, 2, num_anomalies)
    return pd.DataFrame({'x': x, 'y': y})

def save_data():
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    
    for i in range(1, 4):
        data = generate_data(100 + i*10, noise_level=0.1)
        data.to_csv(f'train/data_{i}.csv', index=False)
        
        data = generate_data(50 + i*5, noise_level=0.2, with_anomalies=True)
        data.to_csv(f'test/data_{i}.csv', index=False)

if __name__ == "__main__":
    save_data()