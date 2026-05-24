import numpy as np
from matplotlib import pyplot as plt

from base.Data import DataProvider
from base.Perceptron import Perceptron


def plot_decision_boundary(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    error_indices = np.where(Y_test != y_pred)[0]
    correct_indices = np.where(Y_test == y_pred)[0]
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[correct_indices, 0], X_test[correct_indices, 1],
                c=Y_test[correct_indices], cmap='coolwarm', alpha=0.6, label='Правильно')
    plt.scatter(X_test[error_indices, 0], X_test[error_indices, 1],
                c='black', marker='x', s=100, label='Ошибка')
    w0, w1 = model.w[0], model.w[1]
    b = model.b

    x_range = np.array([X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5])
    y_range = -(w0 * x_range + b) / w1

    plt.plot(x_range, y_range, 'k--', label='Граница решения')
    plt.title('Визуализация ошибок классификации')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True)
    plt.show()


data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
model = Perceptron()
model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.1, batch_size=32)

plot_decision_boundary(model, X_test, Y_test)