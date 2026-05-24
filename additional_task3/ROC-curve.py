import matplotlib.pyplot as plt
import numpy as np

from base.Data import DataProvider
from base.Perceptron import Perceptron


def calculate_roc_curve(y_true, y_scores):
    y_true = np.array(y_true)
    if np.min(y_true) == -1:
        y_true = (y_true + 1) // 2
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    return fpr, tpr


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
model = Perceptron()
model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.1, batch_size=32)

y_scores_test = model.forward(X_test)
fpr, tpr = calculate_roc_curve(Y_test, y_scores_test)
roc_auc = np.trapezoid(tpr, fpr)
plot_roc_curve(fpr, tpr, roc_auc)