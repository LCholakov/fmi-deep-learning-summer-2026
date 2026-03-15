import numpy as np




class SquareModel:

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


def create_dataset_square(start, count):
    dataset = []
    for n in range (start, start + count):
        dataset.append([n, n**2])
    return dataset


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_loss(model: SquareModel, params: np.ndarray, dataset):
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


def train(model: SquareModel, dataset, learning_rate, eps, epochs):
    params = model.get_params_as_vector()
    losses = []
    for _ in range(epochs):
        grad = finite_diff_grad(model, params, dataset, eps)
        params -= learning_rate * grad
        model.load_vector(params)
        losses.append(calculate_loss(model, params, dataset))
    return model, losses


def predict_all(model: SquareModel, dataset):
    return [model.forward(x) for (x, _) in dataset]


def main():

    # rng = np.random.default_rng()
    rng = np.random.default_rng(43)
    dataset_square = create_dataset_square(2, 11)
    model = SquareModel(rng)
    # print(f"model {model.get_params_as_vector()}")

    epochs = 100_000
    learning_rate = 0.02
    eps = 0.001

    model, losses = train(model, dataset_square, learning_rate, eps, epochs)
    predictions = predict_all(model, dataset_square)

    print(f"\tSQUARE MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_loss = calculate_loss(model, model.get_params_as_vector(),
                                dataset_square)
    print(
        f"\tSQUARE Final params: {model.get_params_as_vector()}, Final MSE: {final_loss}\n"
    )

    print("\tSQUARE predictions:")
    for (x, y), p in zip(dataset_square, predictions):
        print(f"({x}) -> y_hat={p:.6f}\t(y={y})")


if __name__ == "__main__":
    main()
