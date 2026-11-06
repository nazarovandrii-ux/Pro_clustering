"""Кластеризація даних на основі алгоритму рою частинок (PSO)."""
import numpy as np


class PSOClustering:
    """Алгоритм кластеризації на основі рою частинок.

    Кожна частинка кодує координати центрів усіх k кластерів.
    Цільова функція - внутрішньокластерна сума квадратів відстаней (SSE).
    """

    def __init__(self, n_clusters=3, n_particles=30, max_iter=100,
                 w_max=0.9, w_min=0.4, c1=1.49, c2=1.49,
                 v_max_ratio=0.2, random_state=None):
        self.n_clusters = n_clusters      # кількість кластерів k
        self.n_particles = n_particles    # розмір рою
        self.max_iter = max_iter          # максимальна кількість ітерацій
        self.w_max = w_max                # початковий коефіцієнт інерції
        self.w_min = w_min                # кінцевий коефіцієнт інерції
        self.c1 = c1                      # когнітивний коефіцієнт
        self.c2 = c2                      # соціальний коефіцієнт
        self.v_max_ratio = v_max_ratio    # обмеження швидкості
        self.random_state = random_state

    def _assign(self, X, centers):
        """Відносить кожен об'єкт до найближчого центру (формула 1.4)."""
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(dist, axis=1), np.min(dist, axis=1)

    def _fitness(self, X, centers):
        """Цільова функція SSE (формула 1.5)."""
        _, dist = self._assign(X, centers)
        return np.sum(dist ** 2)

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n, k = X.shape[0], self.n_clusters

        # Ініціалізація: центри - випадково обрані об'єкти вибірки
        idx = np.array([rng.choice(n, k, replace=False)
                        for _ in range(self.n_particles)])
        pos = X[idx]                      # форма (частинки, k, d)
        vel = np.zeros_like(pos)
        v_max = self.v_max_ratio * (X.max(axis=0) - X.min(axis=0))

        # Найкращі власні положення та глобальне найкраще положення
        pbest = pos.copy()
        pbest_f = np.array([self._fitness(X, p) for p in pos])
        g = pbest[np.argmin(pbest_f)].copy()
        g_f = pbest_f.min()

        self.history_ = []
        for t in range(self.max_iter):
            # Лінійне зменшення коефіцієнта інерції (формула 1.8)
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter
            r1 = rng.random(pos.shape)
            r2 = rng.random(pos.shape)

            # Оновлення швидкості (1.6) та положення (1.7)
            vel = (w * vel + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (g - pos))
            vel = np.clip(vel, -v_max, v_max)
            pos = pos + vel

            # Обчислення пристосованості та оновлення pbest і gbest
            f = np.array([self._fitness(X, p) for p in pos])
            better = f < pbest_f
            pbest[better] = pos[better]
            pbest_f[better] = f[better]
            if pbest_f.min() < g_f:
                g_f = pbest_f.min()
                g = pbest[np.argmin(pbest_f)].copy()
            self.history_.append(g_f)

        self.cluster_centers_ = g
        self.inertia_ = g_f
        self.labels_, _ = self._assign(X, g)
        return self

    def predict(self, X):
        labels, _ = self._assign(X, self.cluster_centers_)
        return labels
