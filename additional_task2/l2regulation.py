import numpy as np
from matplotlib import pyplot as plt

from base.Data import DataProvider
from base.Perceptron import Perceptron
from base.main import print_metrics

np.random.seed(42)
data = DataProvider()
X_train, y_train, X_val, y_val = data.generate_and_prepare()

lambdas = [0.0, 0.01, 0.1, 0.5, 1.0]
results = []

plt.figure(figsize=(14, 5))

for l in lambdas:
    model = Perceptron(loss_type='cross_entropy', l2_lambda=l)
    model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1)

    print_metrics(f"lambda = {l}", y_val, model.predict(X_val))

    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label=f'λ={l}')

    plt.subplot(1, 2, 2)
    plt.plot(model.val_loss_history, label=f'λ={l}')

plt.subplot(1, 2, 1)
plt.title('Train Loss для разных λ')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Val Loss для разных λ')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()