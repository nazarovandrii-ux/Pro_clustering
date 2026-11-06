"""Експерименти для розділу 3: підбір параметрів, порівняння методів, графіки.

Кожен варіант запускається RUNS разів з різними значеннями random_state.
Таблиці виводяться в консоль, рисунки 3.1-3.5 зберігаються в папку results.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.pso_clustering import PSOClustering
from src.metrics import evaluate
from src.visualization import plot_clusters

RUNS = 20
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 11})

# Завантаження та нормалізація всіх наборів даних
DATA = {}
for key, title in [("iris", "Iris"), ("wine", "Wine"),
                   ("breast_cancer", "Breast Cancer")]:
    df, y, k = load_dataset(key)
    DATA[title] = (preprocess(df), y, k)


class HybridPSOKMeans:
    """PSO знаходить центри, K-means уточнює їх."""

    def __init__(self, n_clusters, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X):
        pso = PSOClustering(self.n_clusters,
                            random_state=self.random_state).fit(X)
        km = KMeans(self.n_clusters, init=pso.cluster_centers_,
                    n_init=1).fit(X)
        self.inertia_ = km.inertia_
        self.labels_ = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        return self


def run_many(make_model, X, y):
    """Запускає модель RUNS разів і повертає масиви метрик та часу."""
    rows = []
    for seed in range(RUNS):
        start = time.perf_counter()
        model = make_model(seed).fit(X)
        elapsed = time.perf_counter() - start
        rows.append({**evaluate(X, model, y), "Time": elapsed})
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def print_row(label, res):
    print(f"  {label:22s}"
          f"SSE={res['SSE'].mean():.4f}±{res['SSE'].std():.4f}  "
          f"min={res['SSE'].min():.4f}  max={res['SSE'].max():.4f}  "
          f"Sil={res['Silhouette'].mean():.3f}  "
          f"DB={res['Davies-Bouldin'].mean():.3f}  "
          f"ARI={res['ARI'].mean():.3f}  "
          f"t={res['Time'].mean():.3f}s")


# ---------- 3.1 Підбір параметрів (таблиці 3.1-3.3) ----------
X, y, k = DATA["Iris"]

print("\nТаблиця 3.1 - розмір рою:")
for n in [10, 20, 30, 50]:
    print_row(f"{n} частинок", run_many(
        lambda s: PSOClustering(k, n_particles=n, random_state=s), X, y))

print("\nТаблиця 3.2 - коефіцієнт інерції:")
for label, w1, w2 in [("w = 0.7", 0.7, 0.7), ("w = 0.9", 0.9, 0.9),
                      ("w: 0.9 -> 0.4", 0.9, 0.4)]:
    print_row(label, run_many(
        lambda s: PSOClustering(k, w_max=w1, w_min=w2, random_state=s),
        X, y))

print("\nТаблиця 3.3 - коефіцієнти c1, c2:")
for c1, c2 in [(1.0, 1.0), (1.49, 1.49), (2.0, 2.0), (2.5, 0.5), (0.5, 2.5)]:
    print_row(f"c1 = {c1}, c2 = {c2}", run_many(
        lambda s: PSOClustering(k, c1=c1, c2=c2, random_state=s), X, y))

# Рисунок 3.1 - збіжність для різних розмірів рою
plt.figure(figsize=(7, 4.2))
for n in [10, 30, 50]:
    model = PSOClustering(k, n_particles=n, random_state=1).fit(X)
    plt.plot(range(1, len(model.history_) + 1), model.history_,
             label=f"{n} частинок")
km_sse = KMeans(k, n_init=10, random_state=0).fit(X).inertia_
plt.axhline(km_sse, color="gray", ls="--", lw=1, label="K-means (n_init=10)")
plt.xlabel("Ітерація")
plt.ylabel("SSE")
plt.ylim(km_sse * 0.98, km_sse * 1.18)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig_3_1_convergence_iris.png", dpi=200)
plt.close()

# ---------- 3.2 Візуалізація (рисунки 3.2-3.4) ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
plot_clusters(X, PSOClustering(k, random_state=0).fit(X), "PSO", axes[0])
plot_clusters(X, KMeans(k, n_init=10, random_state=0).fit(X),
              "K-means", axes[1])
plt.tight_layout()
plt.savefig("results/fig_3_2_iris_clusters.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, ["Wine", "Breast Cancer"]):
    Xd, _, kd = DATA[name]
    plot_clusters(Xd, PSOClustering(kd, random_state=0).fit(Xd),
                  f"{name} (PSO)", ax)
plt.tight_layout()
plt.savefig("results/fig_3_3_wine_bc_clusters.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4.2))
for name, (Xd, _, kd) in DATA.items():
    h = np.array(PSOClustering(kd, random_state=0).fit(Xd).history_)
    plt.plot(range(1, len(h) + 1), h / h[-1], label=name)
plt.xlabel("Ітерація")
plt.ylabel("SSE / SSE остаточне")
plt.ylim(0.98, 1.6)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig_3_4_convergence_all.png", dpi=200)
plt.close()

# ---------- 3.3 Порівняння методів (таблиця 3.4, рисунок 3.5) ----------
METHODS = ["PSO", "PSO + K-means", "K-means (1 запуск)", "K-means (n_init=10)"]
mean_sse = {}
for name, (Xd, yd, kd) in DATA.items():
    print(f"\nТаблиця 3.4 - {name}:")
    makers = [
        lambda s: PSOClustering(kd, random_state=s),
        lambda s: HybridPSOKMeans(kd, random_state=s),
        lambda s: KMeans(kd, n_init=1, init="random", random_state=s),
        lambda s: KMeans(kd, n_init=10, random_state=s),
    ]
    mean_sse[name] = {}
    for label, make in zip(METHODS, makers):
        res = run_many(make, Xd, yd)
        print_row(label, res)
        mean_sse[name][label] = res["SSE"].mean()

# Рисунок 3.5 - SSE відносно найкращого варіанта K-means
fig, ax = plt.subplots(figsize=(7, 4.2))
names = list(DATA)
width = 0.2
for i, label in enumerate(METHODS):
    ratio = [mean_sse[n][label] / mean_sse[n]["K-means (n_init=10)"]
             for n in names]
    ax.bar(np.arange(len(names)) + (i - 1.5) * width, ratio, width,
           label=label)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("SSE відносно K-means (n_init=10)")
ax.set_ylim(0.95, 1.2)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig_3_5_sse_relative.png", dpi=200)
plt.close()

print("\nГрафіки збережено в папку results")
