"""Візуалізація результатів кластеризації."""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_clusters(X, model, title, ax=None):
    """Діаграма розсіювання кластерів у просторі двох головних компонент."""
    ax = ax or plt.gca()
    pca = PCA(n_components=2).fit(X)
    points = pca.transform(X)
    centers = pca.transform(model.cluster_centers_)
    ax.scatter(points[:, 0], points[:, 1], c=model.labels_,
               cmap="viridis", s=18)
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
               s=160, edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)


def plot_convergence(history, path):
    """Графік збіжності алгоритму PSO."""
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Ітерація")
    plt.ylabel("SSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
