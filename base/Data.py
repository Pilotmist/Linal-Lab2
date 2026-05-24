import numpy as np
from sklearn.datasets import make_classification


class DataProvider:
    def __init__(self, n_samples=500, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def generate_and_prepare(self):
        X, Y = make_classification(
            n_samples=500,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=42,
            n_clusters_per_class=1
        )

        rng = np.random.default_rng(seed=42)

        indices_0 = np.where(Y == 0)[0]
        indices_1 = np.where(Y == 1)[0]

        rng.shuffle(indices_0)
        rng.shuffle(indices_1)

        split_0 = int(len(indices_0) * 0.7)
        split_1 = int(len(indices_1) * 0.7)

        train_idx = np.concatenate([indices_0[:split_0], indices_1[:split_1]])
        test_idx = np.concatenate([indices_0[split_0:], indices_1[split_1:]])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)

        self.X_train = (X_train - mu) / sigma
        self.X_test = (X_test - mu) / sigma
        self.Y_train = Y_train
        self.Y_test = Y_test
        return self.X_train, self.Y_train, self.X_test, self.Y_test