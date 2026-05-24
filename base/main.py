import numpy as np
import matplotlib.pyplot as plt

from base.Data import DataProvider
from base.Perceptron import Perceptron


def roc_auc_score(y_true, y_scores):
    y_true = np.array(y_true)
    if np.min(y_true) == -1:
        y_true = (y_true + 1) // 2
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_score_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    return np.trapz(tpr, fpr)


def print_metrics(name, y_true, y_pred, y_scores=None):
    y_true_norm = np.array(y_true)
    y_pred_norm = np.array(y_pred)
    if np.min(y_true_norm) == -1:
        y_true_norm = (y_true_norm + 1) // 2
        y_pred_norm = (y_pred_norm + 1) // 2

    tp = np.sum((y_true_norm == 1) & (y_pred_norm == 1))
    tn = np.sum((y_true_norm == 0) & (y_pred_norm == 0))
    fp = np.sum((y_true_norm == 0) & (y_pred_norm == 1))
    fn = np.sum((y_true_norm == 1) & (y_pred_norm == 0))

    acc = (tp + tn) / len(y_true_norm) if len(y_true_norm) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"--- Метрики: {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if y_scores is not None:
        auc = roc_auc_score(y_true, y_scores)
        print(f"ROC-AUC:   {auc:.4f}")
    print("\n")

if __name__ == '__main__':
    data = DataProvider()
    X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
    model = Perceptron()
    model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.1, batch_size=32)

    print_metrics("Обучающая выборка", Y_train, model.predict(X_train))
    print_metrics("Тестовая выборка", Y_test, model.predict(X_test))


    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label='Train Loss', lw=2)
    plt.plot(model.val_loss_history, label='Val Loss', lw=2, linestyle='--')
    plt.title('Кривая обучения (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], c='blue', label='Класс 0', alpha=0.5)
    plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], c='orange', label='Класс 1', alpha=0.5)

    x0_vals = np.array([X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5])
    x1_vals = -(model.w[0] * x0_vals + model.b) / model.w[1]
    plt.plot(x0_vals, x1_vals, c='green', lw=2, label='Граница')

    plt.title('Разделяющая граница')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.xlim(x0_vals)
    plt.ylim(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
