import numpy as np
import matplotlib.pyplot as plt


class Xor:

    def __init__(self, rng):
        # underscore convention to indicate it's private (don't change directly)
        # theta is convention for params, but for now easier for me to read as params
        # 9 params b1, w11, w12, b2, w13, w14, b3, w21, w22 for both hidden layers
        self._params = rng.uniform(-1.0, 1.0, 9)  # 9 params, random init

    def get_params_as_vector(self):
        # copy of params vector to avoid mutating in place.
        return self._params.copy()

    def load_vector(self, updated_params: np.ndarray):
        # apparently if I just assing with "...= updated_params" it's like passing a pointer exposing to mutation
        self._params = updated_params.astype(float, copy=True)

    def forward(self, x1, x2):
        b1, w11, w12, b2, w13, w14, b3, w21, w22 = self._params
        h1 = sigmoid(b1 + w11 * x1 + w12 * x2)  # OR
        h2 = sigmoid(b2 + w13 * x1 + w14 * x2)  # NAND
        y = sigmoid(b3 + w21 * h1 + w22 * h2)
        return y


def create_dataset_and():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def create_dataset_or():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def create_dataset_xor():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


def create_dataset_nand():
    return [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_loss(model: Xor, params: np.ndarray, dataset):
    model.load_vector(params)
    total = 0.0
    iters = 0
    for x1, x2, y in dataset:
        p = model.forward(x1, x2)
        total += (y - p)**2  # square
        iters += 1
    return 0.0 if iters == 0 else float(total / iters)  # mean square


def finite_diff_grad(model, params, dataset, eps):
    g = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        e = np.zeros_like(params)
        e[i] = 1.0
        loss_plus = calculate_loss(model, params + eps * e, dataset)
        loss_minus = calculate_loss(model, params - eps * e, dataset)
        g[i] = (loss_plus - loss_minus) / (2.0 * eps)
    model.load_vector(params)
    return g


def train(model: Xor, dataset, learning_rate, eps, epochs):
    params = model.get_params_as_vector()
    losses = []
    for _ in range(epochs):
        grad = finite_diff_grad(model, params, dataset, eps)
        params -= learning_rate * grad
        model.load_vector(params)
        losses.append(calculate_loss(model, params, dataset))
    return model, losses


def predict_all(model: Xor, dataset):
    return [model.forward(x1, x2) for (x1, x2, _) in dataset]


def main():
    rng = np.random.default_rng()
    dataset_xor = create_dataset_xor()
    model = Xor(rng)

    epochs = 100_000
    learning_rate = 0.01
    eps = 0.001

    model, losses = train(model, dataset_xor, learning_rate, eps, epochs)
    predictions = predict_all(model, dataset_xor)

    print(f"\tXOR MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_loss = calculate_loss(model, model.get_params_as_vector(),
                                dataset_xor)
    print(
        f"\tXOR Final params: {model.get_params_as_vector()}, Final MSE: {final_loss}\n"
    )

    print("\tXOR predictions:")
    for (x1, x2, y), p in zip(dataset_xor, predictions):
        print(f"({x1}, {x2}) -> y_hat={p:.6f}\t(y={y})")


if __name__ == "__main__":
    main()
