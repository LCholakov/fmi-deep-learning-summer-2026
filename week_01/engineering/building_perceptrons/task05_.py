# Q: general form of the two models
# A: We implement a single neuron with two inputs and 2 parameters (aka 2 weights)
# multiply weights by inputs, add products, apply sigmoid to get output in range (0,1).
# Consider this a classification task
# x1, x2 -> x1×w1+x2×w2 -> sigmoid

# Q: What do you notice about the confidence the model has in them?
# at input 0,0 model is at 0.5, basically zero confidence. 
# This applies for both OR and AND. Prediction would be correct 
# if we define classification 1 when >0.5 instead of >=0.5. 

# Looking at the other inputs 01, 10, 11, for OR the model is confident 
# and yields a correct classification.

# Looking at the other inputs 01, 10, 11, for AND the model is 
# completely inconfident as there is no way to solve it.
# There's no combination of 2 weights that will push 1,1 high above
# while 0,1 or 1,0 goes below 0.5.

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def create_or_dataset():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def create_and_dataset():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y, size=2)


def calculate_loss(w, dataset):
    errors_sq = []
    for x1, x2, y in dataset:
        y_hat = sigmoid(w[0] * x1 + w[1] * x2)
        errors_sq.append((y - y_hat)**2)
    return float(np.mean(errors_sq)) if errors_sq else 0.0


def finite_diff_grad(w, dataset, eps):
    g = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        loss_plus = calculate_loss(w_plus, dataset)
        loss_minus = calculate_loss(w_minus, dataset)
        g[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return g


def train(w_init, dataset, learning_rate, eps, epochs):
    w = np.array(w_init, dtype=float)
    for epoch in range(1, epochs + 1):
        # loss_before = calculate_loss(w, dataset)
        # weight_before = w.copy()
        g = finite_diff_grad(w, dataset, eps=eps)
        w -= learning_rate * g
        loss_after = calculate_loss(w, dataset)

        # print(f"epoch {epoch}: grad≈{g} w_before={weight_before} w_after={w}"
        #       f" loss_before={loss_before} loss_after={loss_after}")

        print(f"epoch {epoch}\tweights = {w}\tloss = {loss_after}")
    return w


def predict_all(w, dataset):
    preds = []
    for x1, x2, _ in dataset:
        preds.append(sigmoid(w[0] * x1 + w[1] * x2))
    return preds


def main():

    # my_rng = np.random.default_rng(42)
    my_rng = np.random.default_rng()

    or_dataset = create_or_dataset()
    and_dataset = create_and_dataset()

    w_or = initialize_weights(-1.0, 1.0, my_rng)
    w_and = initialize_weights(-1.0, 1.0, my_rng)

    epochs = 100_000
    learning_rate = 0.05
    eps = 0.001

    print(f"OR LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_w_or = train(w_or, or_dataset, learning_rate, eps=eps, epochs=epochs)
    final_loss_or = calculate_loss(final_w_or, or_dataset)
    print(f"Final w (OR): {final_w_or}, Final MSE (OR): {final_loss_or}")
    preds_or = predict_all(final_w_or, or_dataset)
    print("OR predictions:")
    for (x1, x2, y), p in zip(or_dataset, preds_or):
        print(f"({int(x1)}, {int(x2)}) -> y_hat={p} (y={int(y)})")
    print()

    # OUTPUT 
# OR predictions:
# (0, 0) -> y_hat=0.5 (y=0)
# (0, 1) -> y_hat=0.9854599077749402 (y=1)
# (1, 0) -> y_hat=0.9854654641074299 (y=1)
# (1, 1) -> y_hat=0.9997824327801464 (y=1)

    print(f"AND LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_w_and = train(w_and,
                        and_dataset,
                        learning_rate,
                        eps=eps,
                        epochs=epochs)
    final_loss_and = calculate_loss(final_w_and, and_dataset)
    print(f"Final w (AND): {final_w_and}, Final MSE (AND): {final_loss_and}")
    preds_and = predict_all(final_w_and, and_dataset)
    print("AND predictions:")
    for (x1, x2, y), p in zip(and_dataset, preds_and):
        print(f"({int(x1)}, {int(x2)}) -> y_hat={p} (y={int(y)})")

    # OUTPUT 
# AND predictions:
# (0, 0) -> y_hat=0.5 (y=0)
# (0, 1) -> y_hat=0.500000000000198 (y=0)
# (1, 0) -> y_hat=0.4999999999998497 (y=0)
# (1, 1) -> y_hat=0.5000000000000476 (y=1)



if __name__ == "__main__":
    main()


# Q: What has changed in comparison to the previous task?



import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def create_or_dataset():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def create_and_dataset():
    return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y, size=2)


def calculate_loss(w, dataset):
    errors_sq = []
    for x1, x2, y in dataset:
        y_hat = sigmoid(w[0] * x1 + w[1] * x2)
        errors_sq.append((y - y_hat)**2)
    return float(np.mean(errors_sq)) if errors_sq else 0.0


def finite_diff_grad(w, dataset, eps):
    g = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        loss_plus = calculate_loss(w_plus, dataset)
        loss_minus = calculate_loss(w_minus, dataset)
        g[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return g


def train(w_init, dataset, learning_rate, eps, epochs):
    w = np.array(w_init, dtype=float)
    for epoch in range(1, epochs + 1):
        # loss_before = calculate_loss(w, dataset)
        # weight_before = w.copy()
        g = finite_diff_grad(w, dataset, eps=eps)
        w -= learning_rate * g
        loss_after = calculate_loss(w, dataset)

        # print(f"epoch {epoch}: grad≈{g} w_before={weight_before} w_after={w}"
        #       f" loss_before={loss_before} loss_after={loss_after}")

        print(f"epoch {epoch}\tweights = {w}\tloss = {loss_after}")
    return w


def predict_all(w, dataset):
    preds = []
    for x1, x2, _ in dataset:
        preds.append(sigmoid(w[0] * x1 + w[1] * x2))
    return preds


def main():

    # my_rng = np.random.default_rng(42)
    my_rng = np.random.default_rng()

    or_dataset = create_or_dataset()
    and_dataset = create_and_dataset()

    w_or = initialize_weights(-1.0, 1.0, my_rng)
    w_and = initialize_weights(-1.0, 1.0, my_rng)

    epochs = 100_000
    learning_rate = 0.05
    eps = 0.001

    print(f"OR LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_w_or = train(w_or, or_dataset, learning_rate, eps=eps, epochs=epochs)
    final_loss_or = calculate_loss(final_w_or, or_dataset)
    print(f"Final w (OR): {final_w_or}, Final MSE (OR): {final_loss_or}")
    preds_or = predict_all(final_w_or, or_dataset)
    print("OR predictions:")
    for (x1, x2, y), p in zip(or_dataset, preds_or):
        print(f"({int(x1)}, {int(x2)}) -> y_hat={p} (y={int(y)})")
    print()

    # OUTPUT 
# OR predictions:
# (0, 0) -> y_hat=0.5 (y=0)
# (0, 1) -> y_hat=0.9854599077749402 (y=1)
# (1, 0) -> y_hat=0.9854654641074299 (y=1)
# (1, 1) -> y_hat=0.9997824327801464 (y=1)

    print(f"AND LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_w_and = train(w_and,
                        and_dataset,
                        learning_rate,
                        eps=eps,
                        epochs=epochs)
    final_loss_and = calculate_loss(final_w_and, and_dataset)
    print(f"Final w (AND): {final_w_and}, Final MSE (AND): {final_loss_and}")
    preds_and = predict_all(final_w_and, and_dataset)
    print("AND predictions:")
    for (x1, x2, y), p in zip(and_dataset, preds_and):
        print(f"({int(x1)}, {int(x2)}) -> y_hat={p} (y={int(y)})")

    # OUTPUT 
# AND predictions:
# (0, 0) -> y_hat=0.5 (y=0)
# (0, 1) -> y_hat=0.500000000000198 (y=0)
# (1, 0) -> y_hat=0.4999999999998497 (y=0)
# (1, 1) -> y_hat=0.5000000000000476 (y=1)



if __name__ == "__main__":
    main()
