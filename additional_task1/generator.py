import numpy as np

from base.Perceptron import Perceptron
from matplotlib import pyplot as plt


def generate(data_type='linear', n_samples=500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)

    if data_type == 'linear':
        X = rng.standard_normal((n_samples, 2))
        X[:n_samples // 2] += [2, 2]
        X[n_samples // 2:] += [-2, -2]
        Y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    elif data_type == 'xor':
        X = rng.standard_normal((n_samples, 2))
        Y = (X[:, 0] * X[:, 1] > 0).astype(int)
    elif data_type == 'circle':
        X = rng.standard_normal((n_samples, 2))
        Y = (np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) > 1.2).astype(int)

    if noise > 0:
        mask = rng.random(n_samples) < noise
        Y[mask] = 1 - Y[mask]

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

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, Y_train, X_test, Y_test

results = {}
for dtype in ['linear', 'xor', 'circle']:
    X_train, Y_train, X_test, Y_test = generate(data_type=dtype)
    model = Perceptron()
    model.fit(X_train, Y_train, X_test, Y_test, epochs=50)

    preds = model.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    results[dtype] = accuracy

if __name__ == '__main__':
    print(f"{'Тип данных':<15} | {'Точность (Accuracy)':<20}")
    print("-" * 38)
    for k, v in results.items():
        print(f"{k:<15} | {v:.2%}")

    data_types = ['linear', 'xor', 'circle']

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for i, dtype in enumerate(data_types):
        X_train, Y_train, X_test, Y_test = generate(data_type=dtype, n_samples=600, noise=0.05, seed=42)

        ax_train = axes[i, 0]

        ax_train.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], color='blue', alpha=0.6, label='Class 0')
        ax_train.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], color='red', marker='x', alpha=0.6,
                         label='Class 1')
        ax_train.set_title(f'{dtype.upper()} - Train Split')
        ax_train.set_xlabel('Feature 1')
        ax_train.set_ylabel('Feature 2')
        ax_train.grid(True, linestyle='--', alpha=0.5)
        ax_train.legend()

        ax_test = axes[i, 1]
        ax_test.scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], color='blue', alpha=0.6, label='Class 0')
        ax_test.scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], color='red', marker='x', alpha=0.6,
                        label='Class 1')
        ax_test.set_title(f'{dtype.upper()} - Test Split')
        ax_test.set_xlabel('Feature 1')
        ax_test.set_ylabel('Feature 2')
        ax_test.grid(True, linestyle='--', alpha=0.5)
        ax_test.legend()

    plt.tight_layout()
    plt.show()