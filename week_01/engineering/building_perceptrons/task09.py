# Q: Can you reuse the already created models for the AND and OR gates and just change their datasets?

# Yep, same structure, 2 inputs, expected results are linearly separable so, just pass the NAND dataset.

import numpy as np
import matplotlib.pyplot as plt

def create_dataset_nand():
    return [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y, 3) 

def calculate_loss(weights, dataset):
    b, w1, w2 = weights
    errors_sq = []
    for x1, x2, y in dataset:
        p = sigmoid(b + w1 * x1 + w2 * x2)
        errors_sq.append((y - p)**2)
    return np.mean(errors_sq)

def finite_diff_grad(weights, dataset, eps):
    g = np.zeros_like(weights, dtype=float)
    for i in range(len(weights)):
        e = np.zeros_like(weights); e[i] = 1.0
        loss_plus  = calculate_loss(weights + eps * e, dataset)
        loss_minus = calculate_loss(weights - eps * e, dataset)
        g[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return g

def train(weights, dataset, learning_rate, eps, epochs):
    losses = []
    for _ in range(epochs):
        grad = finite_diff_grad(weights, dataset, eps)
        weights -= learning_rate * grad
        losses.append(calculate_loss(weights, dataset))
    return weights, losses

def predict_all(weights, dataset):
    b, w1, w2 = weights
    preds = []
    for x1, x2, _ in dataset:
        preds.append(sigmoid(b + w1 * x1 + w2 * x2))
    return preds

def main():
    rng = np.random.default_rng()
    dataset = create_dataset_nand()
    weights = initialize_weights(-1.0, 1.0, rng)

    epochs = 100_000
    learning_rate = 0.01
    eps = 0.001

    print(f"\tNAND MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weights, losses = train(weights, dataset, learning_rate, eps, epochs)
    final_loss = calculate_loss(weights, dataset)
    print(f"\tAND Final w: {final_weights}, Final MSE: {final_loss}\n")

    predictions_and = predict_all(final_weights, dataset)
    print("\tAND predictions:")
    for (x1, x2, y), p in zip(dataset, predictions_and):
        print(f"({x1}, {x2}) -> y_hat={p}\t(y={y})")

    # plt.plot(np.arange(1, len(losses) + 1), losses)
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("NAND")
    # plt.show()

if __name__ == "__main__":
    main()