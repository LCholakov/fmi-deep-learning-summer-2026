import numpy as np


def get_my_rng():
    return np.random.default_rng(123)


def get_float_0to1(rng):
    return rng.random()


def get_int_1to6(rng):
    return rng.integers(1, 6 + 1)


def main():
    # test how it works
    # rng1 = get_my_rng()
    # rng2 = get_my_rng()
    # print(rng1.random())
    # print(rng1.random())
    # print(rng2.random())
    # print(rng2.random())

    my_rng = get_my_rng()

    print(f'Random float: {get_float_0to1(my_rng)}')
    print(f'Random integet 1: {get_int_1to6(my_rng)}')
    print(f'Random integer 2: {get_int_1to6(my_rng)}')

    curr_step = 50
    print(f'Before throw step = {curr_step}')
    die_roll = get_int_1to6(my_rng)
    match die_roll:
        case 1 | 2:
            curr_step -= 1
        case 3 | 4 | 5:
            curr_step += 1
        case 6:
            curr_step += get_int_1to6(my_rng)

    print(f'After throw step = {curr_step}')


if __name__ == "__main__":
    main()
