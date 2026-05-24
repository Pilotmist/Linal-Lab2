import numpy as np

from base.Data import DataProvider
from base.Perceptron import Perceptron

def cross_validate_with_stats(X, y, learning_rates, batch_sizes, k=5):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    folds = []
    for i in range(k):
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            fold_accuracies = []

            for X_tr, y_tr, X_val, y_val in folds:
                model = Perceptron(loss_type='cross_entropy')
                model.fit(X_tr, y_tr, epochs=30, lr=lr, batch_size=bs)

                preds = model.predict(X_val)
                acc = np.mean(preds == y_val)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)

            results.append({
                'lr': lr,
                'batch': bs,
                'mean_acc': mean_acc,
                'std_acc': std_acc
            })

    return results

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
lr_candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
batch_size_candidates = [8, 16, 32, 64, 128]

stats = cross_validate_with_stats(X_train, Y_train, lr_candidates, batch_size_candidates, k=5)
top_results = sorted(stats, key=lambda x: x['mean_acc'], reverse=True)[:1]

for res in top_results:
    print(f"LR: {res['lr']}, Batch: {res['batch']} | Acc: {res['mean_acc']:.4f} ± {res['std_acc']:.4f}")