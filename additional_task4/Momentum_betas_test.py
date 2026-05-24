from matplotlib import pyplot as plt

from base.Data import DataProvider
from base.Perceptron import Perceptron

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()

betas = [0.5, 0.9, 0.99]
histories = {}

for beta in betas:
    model = Perceptron(loss_type='cross_entropy')
    model.fit(X_train, Y_train, epochs=50, lr=0.05, momentum=beta)
    histories[beta] = model.train_loss_history

plt.figure(figsize=(10, 6))
for beta, history in histories.items():
    plt.plot(history, label=f'Momentum (beta={beta})')

plt.title('Сравнение скорости сходимости при разных значениях Momentum')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()