# Q: general form of the two models
# A: We implement a perceptron with two inputs and 2 parameters (aka 2 weights)
# multiply weights by inputs, add products.
# Consider this a classification task
# x1, x2 -> x1×w1+x2×w2 -> sigmoid

# Q: What do you notice about the confidence the model has in them?
# A: I'm straight up using the x1×w1+x2×w2 as output for predictions.
# IIUC, this would be the so called "logit". <0 classify as 0, 0=unsure, >0 clasify as 1

# In the case of AND:
# input 0,0 -> 0. This is maxumum lack of confidence. Right on the fence.
# input 0,1 -> ~0.3 Low confidence and wrong
# input 1,0 -> ~0.3 Low confidence and wrong
# input 1,1 -> ~0.67 higher confidence, correct, but not satisfactory.
# I guess we could set a threshold of 0.4.
# Note1: impossible to bring input(0,0) to fall below 0.
# Note2: impossible for the model to find weights that dramatically
# increase y_hat for 1,1, while at the same time reducing below zero
# y_hat for 1,0 and 0,1.

# In the case of OR:
# input 0,0 -> 0. This is maxumum lack of confidence. Right on the fence.
# input 0,1 -> ~0.67 Good confidence and correct
# input 1,0 -> ~0.67 Good confidence and correct
# input 1,1 -> ~1.3 higher confidence, correct.
# Note1: impossible to bring input(0,0) to fall below 0.

# Conclusion: OR can be more-or-less approximated with one neuron and two params.
# AND needs a bit more spice to get it to work convincigly - an additional param
# bias, to shift everything a bit in order to solve
# the issue with multiplying and adding only zeroes.

import numpy as np


def create_dataset_and():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def create_dataset_or():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y, 2)


def calculate_loss(weights, dataset):
    errors_sq = []
    for x1, x2, y in dataset:
        y_hat = weights[0] * x1 + weights[1] * x2
        errors_sq.append((y - y_hat)**2)
    return np.mean(errors_sq)


def finite_diff_grad(weights, dataset, eps):
    # seems like I have been just broadcasting a scalar
    # to both weights. Changed to be a vector.
    # loss_plus = calculate_loss(weights + eps, dataset)
    # loss_minus = calculate_loss(weights - eps, dataset)
    # return (loss_plus - loss_minus) / (2.0 * eps)

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
    predictions = []
    for x1, x2, _ in dataset:
        z = weights[0] * x1 + weights[1] * x2
        predictions.append(z)
    return predictions


def main():
    my_rng = np.random.default_rng()
    dataset_and = create_dataset_and()
    dataset_or = create_dataset_or()
    weights_and = initialize_weights(-1.0, 1.0, my_rng)
    weights_or = initialize_weights(-1.0, 1.0, my_rng)

    epochs = 100_000
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

#   AND predictions:
# (0.0, 0.0) -> y_hat=0.0 (y=0.0)
# (0.0, 1.0) -> y_hat=0.3333333333333243  (y=0.0)
# (1.0, 0.0) -> y_hat=0.3333333333333426  (y=0.0)
# (1.0, 1.0) -> y_hat=0.666666666666667   (y=1.0)

    print(f"\tOR MODEL\tLEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weights_or = train(weights_or, dataset_or, learning_rate, eps,
                             epochs)
    final_loss_or = calculate_loss(final_weights_or, dataset_or)
    print(f"\tOR Final w: {final_weights_or}, Final MSE: {final_loss_or}\n")
    predsictions_or = predict_all(final_weights_or, dataset_or)
    print("OR\tpredictions:")
    for (x1, x2, y), p in zip(dataset_or, predsictions_or):
        print(f"({x1}, {x2}) -> y_hat={p}\t(y={y})")


#   OR predictions:
# (0.0, 0.0) -> y_hat=0.0 (y=0.0)
# (0.0, 1.0) -> y_hat=0.66666666666666    (y=1.0)
# (1.0, 0.0) -> y_hat=0.666666666666673   (y=1.0)
# (1.0, 1.0) -> y_hat=1.333333333333333   (y=1.0)

if __name__ == "__main__":
    main()
