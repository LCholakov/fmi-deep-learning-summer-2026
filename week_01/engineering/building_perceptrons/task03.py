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

        print(f"epoch {epoch}: grad≈{g} w_before={weight_before} w_after={w}" 
              f"loss_before={loss_before} loss_after={loss_after}")
    return w


def main():
    my_rng = np.random.default_rng(42)
    my_dataset = create_dataset(6)
    my_lonely_parameter = initialize_weights(0, 10, my_rng)
    # loss = calculate_loss(my_lonely_parameter, my_dataset)

    print("learning_rate = 0.001")
    final_weight = train(my_lonely_parameter, my_dataset, 
                         learning_rate=0.001, eps=0.001, epochs=10)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w (lr=0.001): {final_weight}, Final MSE: {final_loss}\n")

    print("learning_rate = 0.01")
    final_weight = train(my_lonely_parameter, my_dataset, learning_rate=0.01, eps=0.001, epochs=10)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w (lr=0.01): {final_weight}, Final MSE: {final_loss}\n")

    print("learning_rate = 0.01")
    final_weight = train(my_lonely_parameter, my_dataset, learning_rate=0.06, eps=0.001, epochs=10)
    final_loss = calculate_loss(final_weight, my_dataset)
    print(f"Final w (lr=0.01): {final_weight}, Final MSE: {final_loss}\n") 


if __name__ == "__main__":
    main()
