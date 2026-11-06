"""Попередня обробка даних."""
from sklearn.preprocessing import MinMaxScaler


def preprocess(df, features=None):
    """Перевірка даних і нормалізація min-max (формула 2.1)."""
    if features is not None:
        df = df[features]
    print(f"Пропущених значень: {df.isna().sum().sum()}, "
          f"дублікатів: {df.duplicated().sum()}")
    df = df.dropna()
    return MinMaxScaler().fit_transform(df.values)
