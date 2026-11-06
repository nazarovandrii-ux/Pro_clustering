"""Метрики якості кластеризації."""
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score)


def evaluate(X, model, y_true=None):
    result = {
        "SSE": model.inertia_,
        "Silhouette": silhouette_score(X, model.labels_),
        "Davies-Bouldin": davies_bouldin_score(X, model.labels_),
    }
    if y_true is not None:
        result["ARI"] = adjusted_rand_score(y_true, model.labels_)
    return result
