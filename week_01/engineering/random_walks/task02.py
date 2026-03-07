import numpy as np


def get_my_rng():
    return np.random.default_rng(123)


def get_int_1to6(rng):
    return rng.integers(1, 6 + 1)


def get_next_move(die_roll: int, rng: np.random.Generator):
    match die_roll:
        case 1 | 2:
            return -1
        case 3 | 4 | 5:
            return +1
        case 6:
            return get_int_1to6(rng)
        case _:
            return 0


def main():
    my_rng = get_my_rng()
    starting_step = 0
    np_step_track = np.array([starting_step])

    for i in range(100):
        die_roll = get_int_1to6(my_rng)
        np_step_track = np.append(
            np_step_track, np_step_track[-1] + get_next_move(die_roll, my_rng))

    print(np_step_track.tolist()
          )  # cast to list to match the expected format with commas

    # Do you notice anything unexpected in the output?
    # Not really, it's pretty much what I expected - a bit like tango, alternating
    # forward and backward with a strong bias towards forward due to the
    # 1/3 chance to step back, 2/3 chance to step forward.

    # edit: Just read task03. I guess we can't step down towards the basement


if __name__ == "__main__":
    main()
