from matplotlib import pyplot as plt

from base.Data import DataProvider
from base.Perceptron import Perceptron

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()

model_sgd = Perceptron(loss_type='cross_entropy')
model_sgd.fit(X_train, Y_train, epochs=50, lr=0.05)

model_mom = Perceptron(loss_type='cross_entropy')
model_mom.fit(X_train, Y_train, epochs=50, lr=0.05, momentum=0.9)

plt.plot(model_sgd.train_loss_history, label='SGD')
plt.plot(model_mom.train_loss_history, label='Momentum (0.9)')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.title('Сравнение сходимости SGD и Momentum')
plt.legend()
plt.grid(True)
plt.show()