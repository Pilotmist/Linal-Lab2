import numpy as np
import matplotlib.pyplot as plt
from base.Data import DataProvider
from base.Perceptron import Perceptron

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()

batches = [1, 16, 64, 256]

plt.figure(figsize=(14, 5))
print(f"{'Batch Size':<15} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 53)

for bs in batches:
    np.random.seed(42)
    model = Perceptron()
    model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.1, batch_size=bs)

    train_acc = np.mean(model.predict(X_train) == Y_train)
    test_acc = np.mean(model.predict(X_test) == Y_test)
    print(f"{bs:<15} | {train_acc:<15.4f} | {test_acc:<15.4f}")

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label=f'batch={bs}')
    plt.subplot(1, 2, 2)
    plt.plot(model.val_loss_history, label=f'batch={bs}', linestyle='--')

plt.subplot(1, 2, 1)
plt.title('Train Loss для разных батчей')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Val Loss для разных батчей')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()