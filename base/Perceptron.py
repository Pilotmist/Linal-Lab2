import numpy as np


class Perceptron:
    def __init__(self, loss_type='cross_entropy', l2_lambda=0.0):
        self.w = None
        self.b = None
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        if self.loss_type == 'hinge':
            return z
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        if self.loss_type == 'cross_entropy':
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            if self.l2_lambda > 0 and self.w is not None:
                loss += (self.l2_lambda / 2) * np.sum(self.w**2)
            return loss
        else:
            losses = np.maximum(0, 1 - y_true * y_pred)
            return np.mean(losses)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.1,
            batch_size=32, momentum=0.0, init_type='small_random'):
        n_samples, n_features = X_train.shape
        if init_type == 'zeros':
            self.w = np.zeros(n_features)
        elif init_type == 'large_random':
            self.w = np.random.randn(n_features) * 10.0
        else:
            self.w = np.random.randn(n_features) * 0.01

        self.b = 0.0
        v_w = np.zeros_like(self.w)
        v_b = 0.0

        self.train_loss_history = []
        self.val_loss_history = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                m_batch = X_batch.shape[0]

                y_pred = self.forward(X_batch)

                if self.loss_type == 'cross_entropy':
                    dw = np.dot(X_batch.T, (y_pred - y_batch)) / m_batch
                    db = np.sum(y_pred - y_batch) / m_batch
                    if self.l2_lambda > 0:
                        dw += self.l2_lambda * self.w
                else:
                    mask = (y_batch * y_pred) < 1
                    if np.any(mask):
                        dw = -np.dot(X_batch[mask].T, y_batch[mask]) / m_batch
                        db = -np.sum(y_batch[mask]) / m_batch
                    else:
                        dw = np.zeros_like(self.w)
                        db = 0.0
                if momentum > 0:
                    v_w = momentum * v_w + lr * dw
                    v_b = momentum * v_b + lr * db
                    self.w -= v_w
                    self.b -= v_b
                else:
                    self.w -= lr * dw
                    self.b -= lr * db
            self.train_loss_history.append(self.compute_loss(y_train, self.forward(X_train)))
            if X_val is not None and y_val is not None:
                self.val_loss_history.append(self.compute_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        if self.loss_type == 'cross_entropy':
            return (self.forward(X) >= 0.5).astype(int)
        else:
            return np.where(self.forward(X) >= 0, 1, -1)