import numpy as np


def create_dataset(n: int):
    return_list = []
    for i in range(n):
        return_list.append([i, 2 * i])
    # print(return_list)
    return return_list


def initialize_weights(x, y, rng: np.random.Generator):
    return rng.uniform(x, y)


def calculate_loss(w, dataset):
    errors_sq = []
    for x, y in dataset:
        y_hat = w * x
        errors_sq.append((y - y_hat)**2)
    return float(np.mean(errors_sq)) if errors_sq else 0.0


def finite_diff_grad(w, dataset, eps):
    loss_plus = calculate_loss(w + eps, dataset)
    loss_minus = calculate_loss(w - eps, dataset)
    return (loss_plus - loss_minus) / (2.0 * eps)


def train(w_init, dataset, learning_rate, eps, epochs):
    w = float(w_init)
    for epoch in range(1, epochs + 1):
        loss_before = calculate_loss(w, dataset)
        weight_before = w
        g = finite_diff_grad(w, dataset, eps=eps)
        w -= learning_rate * g
        loss_after = calculate_loss(w, dataset)

        # print(f"epoch {epoch}: grad≈{g} w_before={weight_before} w_after={w}"
        #   f"loss_before={loss_before} loss_after={loss_after}")
    return w


def main():
    # my_rng = np.random.default_rng(42)
    my_rng = np.random.default_rng()
    my_dataset = create_dataset(5)
    my_lonely_parameter = initialize_weights(0, 10, my_rng)
    # loss = calculate_loss(my_lonely_parameter, my_dataset)
    epochs = 10
    learning_rate = 0.001
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")

    learning_rate = 0.01
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")

    learning_rate = 0.06
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")

    epochs = 500
    learning_rate = 0.001
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")

    learning_rate = 0.01
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")

    learning_rate = 0.06
    print(f"LEARNING RATE = {learning_rate}\tEPOCHS = {epochs}")
    final_weight = train(my_lonely_parameter,
                         my_dataset,
                         learning_rate,
                         eps=0.001,
                         epochs=epochs)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w: {final_weight}, Final MSE: {final_loss}\n")


# This would be 100x more readable if I arranged these in a table. Some other time.

# SEED = 42
#
# LEARNING RATE = 0.001   EPOCHS = 10
# Final w: 7.086840060908497, Final MSE: 155.2556508315814

# LEARNING RATE = 0.01    EPOCHS = 10
# Final w: 3.5984731970929085, Final MSE: 15.330699370946547

# LEARNING RATE = 0.06    EPOCHS = 10
# Final w: 2.000017000392581, Final MSE: 1.734080087405969e-09

# LEARNING RATE = 0.001   EPOCHS = 500
# Final w: 2.0137198995218086, Final MSE: 0.0011294138573311556

# LEARNING RATE = 0.01    EPOCHS = 500
# Final w: 2.0000000000000018, Final MSE: 1.8932661725304283e-29

# LEARNING RATE = 0.06    EPOCHS = 500
# Final w: 2.0, Final MSE: 0.0

# RANDOM SEED
#
# LEARNING RATE = 0.001   EPOCHS = 10
# Final w: 4.5328628795851165, Final MSE: 38.492366200681246

# LEARNING RATE = 0.01    EPOCHS = 10
# Final w: 2.7959191514675883, Final MSE: 3.8009237740373143

# LEARNING RATE = 0.06    EPOCHS = 10
# Final w: 2.000008464913933, Final MSE: 4.299286073544774e-10

# LEARNING RATE = 0.001   EPOCHS = 500
# Final w: 2.006831475689102, Final MSE: 0.0002800143605447395

# LEARNING RATE = 0.01    EPOCHS = 500
# Final w: 2.0000000000000018, Final MSE: 1.8932661725304283e-29

# LEARNING RATE = 0.06    EPOCHS = 500
# Final w: 2.0, Final MSE: 0.0

if __name__ == "__main__":
    main()
