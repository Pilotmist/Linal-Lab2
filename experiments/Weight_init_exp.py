import numpy as np
import matplotlib.pyplot as plt
from base.Data import DataProvider
from base.Perceptron import Perceptron

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()

initializations = ['zeros', 'small_random', 'large_random']

plt.figure(figsize=(14, 5))
print(f"{'Initialization':<15} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 53)

for init in initializations:
    np.random.seed(42)
    model = Perceptron()
    model.fit(X_train, Y_train, X_test, Y_test, epochs=100, lr=0.1, batch_size=32, init_type=init)

    train_acc = np.mean(model.predict(X_train) == Y_train)
    test_acc = np.mean(model.predict(X_test) == Y_test)
    print(f"{init:<15} | {train_acc:<15.4f} | {test_acc:<15.4f}")

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label=f'{init}')
    plt.subplot(1, 2, 2)
    plt.plot(model.val_loss_history, label=f'{init}', linestyle='--')

plt.subplot(1, 2, 1)
plt.title('Train Loss для разных инициализаций')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Val Loss для разных инициализаций')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()