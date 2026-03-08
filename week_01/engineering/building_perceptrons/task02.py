import numpy as np

def create_dataset(n: int):
    return_list = []
    for i in range(n):
        return_list.append([i, 2 * i])
    # print(return_list)
    return return_list

def initialize_weights(x, y, rng:np.random.Generator):
    return rng.uniform(x, y)

def calculate_loss(w, dataset):
    # mse explanation and snippet from copilot. 
    errors_sq = []
    for x, y in dataset:
        y_hat = w * x
        errors_sq.append((y - y_hat) ** 2)
    return float(np.mean(errors_sq)) if errors_sq else 0.0

    

def main():
    my_rng = np.random.default_rng(42)
    my_dataset = create_dataset(6)
    my_lonely_parameter = initialize_weights(0,10, my_rng)
    loss = calculate_loss(my_lonely_parameter, my_dataset)
    print(f'MSE: {loss}')
    # with seed 42, always getting 301.9734168678107. 
    # I would think it should be the same as the expected result 27.92556532998047 
    # since the seed is the same 42


# Q: What happens to loss function when you pass w + 0.001 * 2, w + 0.001, w - 0.001 and w - 0.001 * 2?
    # print(calculate_loss(my_lonely_parameter + 0.001, my_dataset))
    # print(calculate_loss(my_lonely_parameter + 0.002, my_dataset))
    # print(calculate_loss(my_lonely_parameter - 0.001, my_dataset))
    # print(calculate_loss(my_lonely_parameter - 0.002, my_dataset))
# A: adding small step increases the loss 301.97 -> 302.08 -> 302.18
# subtracting a small step, i.e. epsilon, decreases the loss 301.97 -> 301.87 -> 301.76
# larger step increases the amount of loss change, but the direction stays the same. 

if __name__ == "__main__":
    main()