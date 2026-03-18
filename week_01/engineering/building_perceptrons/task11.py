# This needs to result in an approximation, not exact x*x
# There actually are some NN architectures multiplying f(x) with g(x), 
# most notably LSTMs and Highway networks. But even these have one or 
# both of f(x), g(s) bounded (by logistic sigmoid or tanh), thus are 
# unable to model x*x fully.
# ref: https://stackoverflow.com/questions/55170460/neural-network-for-square-x2-approximation


import numpy as np


class ModelPowerOfTwo:

    def __init__(self, rng):
        self._params = rng.uniform(-1.0, 1.0, 10)

    def get_params_as_vector(self):
        return self._params.copy()

    def load_vector(self, updated_params: np.ndarray):
        self._params = updated_params.astype(float, copy=True)

    def forward(self, x):
        b1, w11, b2, w12, b3, w13, b4, w21, w22, w23 = self._params
        h1 = sigmoid(b1 + w11 * x)
        h2 = sigmoid(b2 + w12 * x)
        h3 = sigmoid(b3 + w13 * x)
        y = b4 + w21 * h1 + w22 * h2 + w23 * h3
        return y


def create_dataset_power_of_two(start, count):
    dataset = []
    for n in range (start, start + count):
        dataset.append([n, n**2])
    return dataset


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_loss(model: ModelPowerOfTwo, params: np.ndarray, dataset):
    model.load_vector(params)
    total = 0.0
    iters = 0
    for x, y in dataset:
        p = model.forward(x)
        total += (y - p)**2
        iters += 1
    return 0.0 if iters == 0 else float(total / iters) 


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


def train(model: ModelPowerOfTwo, dataset, learning_rate, eps, epochs):
    params = model.get_params_as_vector()
    losses = []
    for _ in range(epochs):
        grad = finite_diff_grad(model, params, dataset, eps)
        params -= learning_rate * grad
        model.load_vector(params)
        losses.append(calculate_loss(model, params, dataset))
    return model, losses


def predict_all(model: ModelPowerOfTwo, dataset):
    return [model.forward(x) for (x, _) in dataset]


def main():

    # rng = np.random.default_rng()
    rng = np.random.default_rng(43)
    dataset_square = create_dataset_power_of_two(2, 11)
    model = ModelPowerOfTwo(rng)
    # print(f"model {model.get_params_as_vector()}")

    epochs = 100_000
    learning_rate = 0.02
    eps = 0.001

    model, losses = train(model, dataset_square, learning_rate, eps, epochs)
    predictions = predict_all(model, dataset_square)

    print(f"\tMODEL POWER OF TWO\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_loss = calculate_loss(model, model.get_params_as_vector(),
                                dataset_square)
    print(
        f"\tPOWER OF TWO Final params: {model.get_params_as_vector()}, Final MSE: {final_loss}\n"
    )

    print("\tPOWER OF TWO predictions:")
    for (x, y), p in zip(dataset_square, predictions):
        print(f"({x}) -> y_hat={p:.6f}\t(y={y})")


if __name__ == "__main__":
    main()

#         MODEL POWER OF TWO      LEARNING RATE = 0.02    EPOCHS = 100000
#         POWER OF TWO Final params: [ 5.50052472  3.50038085  0.78625983 25.30856546  7.16053282  2.85733179     
#  32.86716236 -2.493431   32.05915147 -3.43287639], Final MSE: 2038.0001881508529

#         POWER OF TWO predictions:
# (2) -> y_hat=59.000025  (y=4)
# (3) -> y_hat=59.000007  (y=9)
# (4) -> y_hat=59.000006  (y=16)
# (5) -> y_hat=59.000006  (y=25)
# (6) -> y_hat=59.000006  (y=36)
# (7) -> y_hat=59.000006  (y=49)
# (8) -> y_hat=59.000006  (y=64)
# (9) -> y_hat=59.000006  (y=81)
# (10) -> y_hat=59.000006 (y=100)
# (11) -> y_hat=59.000006 (y=121)
# (12) -> y_hat=59.000006 (y=144)

# Why this happened


# Sigmoid saturation + unscaled inputs/targets.
# With inputs in [2 … 12] and random weights in [-1, 1], 
# the pre-activations b + w*x can easily reach large 
# magnitudes (e.g., |z| ≳ 5), pushing the three sigmoids 
# to ~0 or ~1 almost everywhere. Once hidden units act like 
# constants, the model reduces to y ≈ constant, and the MSE 
# optimum becomes the mean.
# Your final weights also reflect this: because hidden units 
# saturated, the output learned b4 + w21 + w22 + w23 ≈ 59. 

# Finite-difference gradients on a flat landscape.
# With saturated sigmoids, the loss surface gets very flat w.r.t. 
# hidden-layer parameters. Finite differences then report near‑zero 
# change for most directions, so the only effective move is to adjust 
# the output bias/weights to the mean.

