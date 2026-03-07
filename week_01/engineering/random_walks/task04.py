import numpy as np
import matplotlib.pyplot as plt

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
        next_pos = np_step_track[-1] + get_next_move(die_roll, my_rng)
        if(next_pos < 0):
            np_step_track = np.append(np_step_track, np_step_track[-1])
        else: 
            np_step_track = np.append(np_step_track, next_pos)

    fig, ax = plt.subplots(figsize=(6, 4))

    
    # My chart looks different from expected.
    # compared my output data with data with expected resykt  => expected same as mine.
    # expected = [0, 0, 1, 2, 1, 3, 2, 1, 2, 1, 2, 3, 4, 7, 6, 7, 8, 14, 13, 14, 13, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 20, 19, 24, 23, 24, 25, 24, 23, 22, 21, 22, 21, 22, 23, 22, 23, 22, 21, 22, 21, 22, 21, 23, 24, 25, 29, 32, 31, 30, 32, 33, 34, 33, 34, 35, 36, 42, 41, 42, 43, 44, 43, 44, 45, 47, 48, 49, 51, 52, 51, 52, 51, 52, 51, 52, 53, 54, 55, 56, 55, 54, 55, 54, 55, 56, 57, 58, 57, 58, 59]
    print(np_step_track.tolist())
    # my ouput = [0, 0, 1, 2, 1, 3, 2, 1, 2, 1, 2, 3, 4, 7, 6, 7, 8, 14, 13, 14, 13, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 20, 19, 24, 23, 24, 25, 24, 23, 22, 21, 22, 21, 22, 23, 22, 23, 22, 21, 22, 21, 22, 21, 23, 24, 25, 29, 32, 31, 30, 32, 33, 34, 33, 34, 35, 36, 42, 41, 42, 43, 44, 43, 44, 45, 47, 48, 49, 51, 52, 51, 52, 51, 52, 51, 52, 53, 54, 55, 56, 55, 54, 55, 54, 55, 56, 57, 58, 57, 58, 59]
    # ax.plot(expected)
    # the img in Task 04 shows a big drop around iteration #5
    ax.plot(np_step_track)

    plt.show()

if __name__ == "__main__":
    main()
