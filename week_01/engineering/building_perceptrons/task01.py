import numpy as np

def create_dataset(n: int):
    return_list = []
    for i in range(n):
        return_list.append([i, 2 * i])
    # print(return_list)
    return return_list

def initialize_weights(x, y):
    return np.random.uniform(x, y)

def main():
    print(create_dataset(4))
    print(initialize_weights(0,100))
    print(initialize_weights(0,10))

if __name__ == "__main__":
    main()