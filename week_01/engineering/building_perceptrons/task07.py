import numpy as np
import matplotlib.pyplot as plt


def create_dataset_and():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def create_dataset_or():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y, 3)


def calculate_loss(weights, dataset):
    bias = weights[0]
    errors_sq = []
    for x1, x2, y in dataset:
        y_hat = bias + weights[1] * x1 + weights[2] * x2
        errors_sq.append((y - y_hat)**2)
    return np.mean(errors_sq)


def finite_diff_grad(weights, dataset, eps):
    g = np.zeros_like(weights, dtype=float)
    for i in range(len(weights)):
        e = np.zeros_like(weights)
        e[i] = 1.0
        loss_plus = calculate_loss(weights + eps * e, dataset)
        loss_minus = calculate_loss(weights - eps * e, dataset)
        g[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return g


def train(weights, dataset, learning_rate, eps, epochs):
    for epoch in range(1, epochs + 1):
        grad = finite_diff_grad(weights, dataset, eps)
        weights -= learning_rate * grad
        loss = calculate_loss(weights, dataset)
        print(f"epoch {epoch}\tweights = {weights}\tloss = {loss}")
    return weights


def predict_all(weights, dataset):
    bias = weights[0]
    predictions = []
    for x1, x2, _ in dataset:
        z = bias + weights[1] * x1 + weights[2] * x2
        predictions.append(z)
    return predictions


def main():
    my_rng = np.random.default_rng()
    dataset_and = create_dataset_and()
    dataset_or = create_dataset_or()
    weights_and = initialize_weights(-1.0, 1.0, my_rng)
    weights_or = initialize_weights(-1.0, 1.0, my_rng)

    epochs = 100_000  #converges pretty quickly, need another stop criteria, eg when change stops
    learning_rate = 0.01
    eps = 0.001

    z = np.linspace(-10, 10, 1000)
    plt.plot(z, sigmoid(z))
    plt.show()


if __name__ == "__main__":
    main()
