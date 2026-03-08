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
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))


# Q: A general form of the model is placed in a comment that shows how many parameters the model has.
# A: Each training sample has one input and one expected output.
# The model will have only one parameter (weight w). Weight will modify the input.
# We'll then compare the output to the expected result and calculate how to change
# the weight.
#
# ※ Note on bias - ignoring bias for this model, but the general model would require
# a bias as well. This would bring the parameter count up to 2, but in this
# case I think bias would be set to zero, so I'm just skipping it.

if __name__ == "__main__":
    main()
