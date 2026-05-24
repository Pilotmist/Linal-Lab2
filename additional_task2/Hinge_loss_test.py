import numpy as np

from base.Data import DataProvider
from base.Perceptron import Perceptron
from base.main import print_metrics

data = DataProvider()
X_train, Y_train, X_test, Y_test = data.generate_and_prepare()
model_ce = Perceptron(loss_type='cross_entropy')
model_ce.fit(X_train, Y_train)
pred_ce = model_ce.predict(X_test)
print_metrics("Cross-Entropy", Y_test, pred_ce)

Y_train_h = np.where(Y_train == 0, -1, 1)
model_h = Perceptron(loss_type='hinge')
model_h.fit(X_train, Y_train_h)
pred_h = model_h.predict(X_test)
print_metrics("Hinge Loss", Y_test, pred_h)

