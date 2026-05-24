import numpy as np
import matplotlib.pyplot as plt
from base.Data import DataProvider
from base.Perceptron import Perceptron

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
lrs = [0.001, 0.01, 0.5, 1.0]
plt.figure(figsize=(14, 5))
print(f"{'Learning Rate':<15} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 53)

for lr in lrs:
    np.random.seed(42)
    model = Perceptron()
    model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=lr, batch_size=32)

    train_acc = np.mean(model.predict(X_train) == Y_train)
    test_acc = np.mean(model.predict(X_test) == Y_test)
    print(f"{lr::<15} | {train_acc:<15.4f} | {test_acc:<15.4f}")

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label=f'lr={lr}')
    plt.subplot(1, 2, 2)
    plt.plot(model.val_loss_history, label=f'lr={lr}', linestyle='--')

plt.subplot(1, 2, 1)
plt.title('Train Loss для разных lr')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Val Loss для разных lr')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()