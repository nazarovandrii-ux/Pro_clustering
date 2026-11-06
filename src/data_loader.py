"""Завантаження наборів даних."""
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

DATASETS = {
    "iris": (load_iris, 3),
    "wine": (load_wine, 3),
    "breast_cancer": (load_breast_cancer, 2),
}


def load_dataset(name):
    """Повертає таблицю ознак, справжні мітки класів і кількість кластерів."""
    loader, k = DATASETS[name]
    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data.target, k


def load_csv(path):
    """Завантаження власного набору даних із файлу CSV."""
    return pd.read_csv(path)
