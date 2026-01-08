import numpy as np

# -----------------------------
# Dataset (Binary Classification)
# -----------------------------
X = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [1, 50],
    [9, 95]
], dtype=float)

y = np.array([[0], [0], [1], [1], [0], [1]])

# Normalize features
X = X / np.max(X, axis=0)


# -----------------------------
# ANN From Scratch
# -----------------------------
class ANNFromScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.05):
        np.random.seed(42)
        self.lr = lr

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    # Activation functions
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # -----------------------------
    # Forward Propagation
    # -----------------------------
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self.sigmoid(self.z2)

        return self.y_pred

    # -----------------------------
    # Binary Cross-Entropy Loss
    # -----------------------------
    def compute_loss(self, y, y_pred):
        eps = 1e-8  # avoid log(0)
        return -np.mean(
            y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
        )

    # -----------------------------
    # Backpropagation
    # -----------------------------
    def backward(self, X, y):
        m = X.shape[0]

        # Output layer gradient
        dz2 = self.y_pred - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient Descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # -----------------------------
    # Training Loop
    # -----------------------------
    def train(self, X, y, epochs=3000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)

            if epoch % 500 == 0:
                acc = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")


# -----------------------------
# Train Model
# -----------------------------
model = ANNFromScratch(input_dim=2, hidden_dim=4, output_dim=1)
model.train(X, y)

# Predictions
preds = model.forward(X)
print("\nFinal Predictions:")
print(np.round(preds, 3))
print("Predicted Classes:", (preds > 0.5).astype(int).ravel())
