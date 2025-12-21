import numpy as np
from dataclasses import dataclass
from typing import Tuple



@dataclass
class Config:
    input_dim: int = 2
    hidden_dim: int = 4
    output_dim: int = 1
    learning_rate: float = 0.1
    epochs: int = 10000
    random_state: int = 42


cfg = Config()
np.random.seed(cfg.random_state)

class Sigmoid:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Numerical stability
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(output: np.ndarray) -> np.ndarray:
        return output * (1 - output)


class MeanSquaredError:
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_true - y_pred)



class DenseLayer:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros((1, out_features))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        grad_W = np.dot(self.input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.W.T)

        self.W += lr * grad_W
        self.b += lr * grad_b

        return grad_input


class NeuralNetwork:
    def __init__(self, cfg: Config):
        self.fc1 = DenseLayer(cfg.input_dim, cfg.hidden_dim)
        self.fc2 = DenseLayer(cfg.hidden_dim, cfg.output_dim)
        self.activation = Sigmoid()
        self.loss_fn = MeanSquaredError()
        self.lr = cfg.learning_rate

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = self.fc1.forward(X)
        self.a1 = self.activation.forward(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        return self.activation.forward(self.z2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        loss_grad = self.loss_fn.backward(y_true, y_pred)

        d_output = loss_grad * self.activation.backward(y_pred)
        d_hidden = self.fc2.backward(d_output, self.lr)

        d_hidden *= self.activation.backward(self.a1)
        self.fc1.backward(d_hidden, self.lr)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn.forward(y, y_pred)
            self.backward(y, y_pred)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.6f}")



X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])


model = NeuralNetwork(cfg)
model.train(X, y, cfg.epochs)

print("\nFinal Predictions:")
preds = model.forward(X)
print(np.round(preds, 3))
