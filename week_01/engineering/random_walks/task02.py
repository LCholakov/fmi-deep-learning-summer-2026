import numpy as np

def get_my_rng():
    return np.random.default_rng(123)

def get_int_1to6(rng):
    return rng.integers(1, 6+1)

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
        np_step_track = np.append(np_step_track, np_step_track[-1] + get_next_move(die_roll, my_rng))

    print(np_step_track)

    # Do you notice anything unexpected in the output?

if __name__ == "__main__":
    main()
