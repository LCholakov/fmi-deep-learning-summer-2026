# Q: What has changed in comparison to the previous task?
# A: I feel I'm doing sth incorrect. I was expecting 
# negative values for AND inputs 1,0 1,0. 


import numpy as np


def create_dataset_and():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def create_dataset_or():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


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

    epochs = 100_000 #converges pretty quickly, need another stop criteria, eg when change stops 
    learning_rate = 0.01
    eps = 0.001

    print(f"\tAND MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weights_and = train(weights_and, dataset_and, learning_rate, eps,
                              epochs)
    final_loss_and = calculate_loss(final_weights_and, dataset_and)
    print(f"\tAND Final w: {final_weights_and}, Final MSE: {final_loss_and}\n")
    predsictions_and = predict_all(final_weights_and, dataset_and)
    print("\tAND predictions:")
    for (x1, x2, y), p in zip(dataset_and, predsictions_and):
        print(f"({x1}, {x2}) -> y_hat={p}\t(y={y})")


#   AND predictions without bias:
# (0.0, 0.0) -> y_hat=0.0 (y=0.0)
# (0.0, 1.0) -> y_hat=0.3333333333333243  (y=0.0)
# (1.0, 0.0) -> y_hat=0.3333333333333426  (y=0.0)
# (1.0, 1.0) -> y_hat=0.666666666666667   (y=1.0)

#   AND predictions with bias:
# (0.0, 0.0) -> y_hat=-0.25000000000001155        (y=0.0)
# (0.0, 1.0) -> y_hat=0.249999999999997   (y=0.0)
# (1.0, 0.0) -> y_hat=0.24999999999999545 (y=0.0)
# (1.0, 1.0) -> y_hat=0.750000000000004   (y=1.0)

    print(f"\tOR MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weights_or = train(weights_or, dataset_or, learning_rate, eps,
                            epochs)
    final_loss_or = calculate_loss(final_weights_or, dataset_or)
    print(f"\tOR Final w: {final_weights_or}, Final MSE: {final_loss_or}\n")
    predsictions_or = predict_all(final_weights_or, dataset_or)
    print("\tpredictions:")
    for (x1, x2, y), p in zip(dataset_or, predsictions_or):
        print(f"({x1}, {x2}) -> y_hat={p}\t(y={y})")

#   OR predictions without bias:
# (0.0, 0.0) -> y_hat=0.0 (y=0.0)
# (0.0, 1.0) -> y_hat=0.66666666666666    (y=1.0)
# (1.0, 0.0) -> y_hat=0.666666666666673   (y=1.0)
# (1.0, 1.0) -> y_hat=1.333333333333333   (y=1.0)

#   OR predictions with bias:
# (0.0, 0.0) -> y_hat=0.25000000000002115 (y=0.0)
# (0.0, 1.0) -> y_hat=0.7500000000000022  (y=1.0)
# (1.0, 0.0) -> y_hat=0.7500000000000027  (y=1.0)
# (1.0, 1.0) -> y_hat=1.2499999999999838  (y=1.0)

if __name__ == "__main__":
    main()
