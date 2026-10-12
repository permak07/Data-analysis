import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Функция для чтения файла, возвращает массив
def load_dataset(filepath):
    d = []
    with open(filepath, 'r', encoding='utf-8') as f:
        is_data = False
        for s in f:
            if 'Freq' in s and 'Range' in s:
                is_data = True
                continue
            if is_data:
                val = s.split()
                if len(val) >= 2:
                    try:
                        d.append([float(val[0]), float(val[1])])
                    except ValueError:
                        continue
    return np.array(d)

# Фиункция кластеризации
def cluster_data(filepath):
    print(f"Кластеризация: {os.path.basename(filepath)}")
    
    d = load_dataset(filepath)
    
    # масштабируем координаты
    scaler = StandardScaler()
    X = scaler.fit_transform(d)
    
    # Кластеризуем с помощью KMeans c n=6, так как это лучший вариант кластеризации по исследованию
    clf = KMeans(n_clusters=6, n_init=10, random_state=42)
    c = clf.fit_predict(X)
    
    # Таблица с номерами кластеров
    res = pd.DataFrame({
        'Freq': d[:, 0],
        'Range': d[:, 1],
        'Cluster': c
    })
    
    # График
    fig = plt.figure()
    plt.scatter(res['Freq'], res['Range'], c=res['Cluster'], cmap='tab10', s=15)
    plt.xlabel("Freq")
    plt.ylabel("Range")
    plt.show()
        
    return res

if __name__ == "__main__":
    folder = "/home/permak07/Desktop/Education/subjects 2 curs/Data Analysis/dz/Clustering/2test"
    count=0
    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            res=cluster_data(filepath)
            if count<3:
                print(f"Первые 5 строк файла: {filename}")
                print(res.head(5), '\n')
                count+=1