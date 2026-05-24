from base.Data import DataProvider
from base.Perceptron import Perceptron
from additional_task1.generator import generate

import numpy as np

results_data = []
datasets = ['linear', 'xor', 'circle', 'DataProvider']

for dtype in datasets:
    if dtype == 'DataProvider':
        provider = DataProvider()
        X_tr, Y_tr, X_te, Y_te = provider.generate_and_prepare()
    else:
        X_tr, Y_tr, X_te, Y_te = generate(data_type=dtype)

    model = Perceptron(loss_type='cross_entropy', l2_lambda=0.01)
    model.fit(X_tr, Y_tr, X_te, Y_te, epochs=100, lr=0.05, batch_size=16, momentum=0.9)

    y_pred = model.predict(X_te)

    tp = np.sum((Y_te == 1) & (y_pred == 1))
    tn = np.sum((Y_te == 0) & (y_pred == 0))
    fp = np.sum((Y_te == 0) & (y_pred == 1))
    fn = np.sum((Y_te == 1) & (y_pred == 0))

    acc = (tp + tn) / len(Y_te)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    results_data.append((dtype, acc, prec, rec, f1))

print(f"{'Dataset':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 75)
for res in results_data:
    print(f"{res[0]:<20} | {res[1]:<10.4f} | {res[2]:<10.4f} | {res[3]:<10.4f} | {res[4]:<10.4f}")