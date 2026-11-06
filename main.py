"""Точка входу: кластеризація даних методами PSO та K-means."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.pso_clustering import PSOClustering
from src.metrics import evaluate
from src.visualization import plot_clusters, plot_convergence

os.makedirs("results", exist_ok=True)

for name in ["iris", "wine", "breast_cancer"]:
    df, y, k = load_dataset(name)
    X = preprocess(df)

    pso = PSOClustering(n_clusters=k, n_particles=30, max_iter=100,
                        random_state=42).fit(X)
    hybrid = KMeans(n_clusters=k, init=pso.cluster_centers_,
                    n_init=1).fit(X)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)

    print(f"\n=== {name} ===")
    for label, model in [("PSO", pso), ("PSO + K-means", hybrid),
                         ("K-means", kmeans)]:
        m = evaluate(X, model, y)
        print(f"{label:15s}", "  ".join(f"{k_}={v:.4f}" for k_, v in m.items()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_clusters(X, pso, "PSO", axes[0])
    plot_clusters(X, kmeans, "K-means", axes[1])
    plt.tight_layout()
    plt.savefig(f"results/{name}_clusters.png", dpi=200)
    plt.close()
    plot_convergence(pso.history_, f"results/{name}_convergence.png")
