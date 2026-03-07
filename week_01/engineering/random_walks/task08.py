import numpy as np
import matplotlib.pyplot as plt


def get_my_rng():
    return np.random.default_rng(123)


def get_float_0to1(rng):
    return rng.random()


def get_int_1to6(rng):
    #.item suggested by copilot to get plain int, because otherwise my output with lists
    # showed 0, 0, 1, 2, 1, np.int64(3), np.int64(2), np.int64(1)...
    return rng.integers(1, 6 + 1).item()


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


def get_one_walk(rng: np.random.Generator):
    starting_step = 0
    step_track = [starting_step]
    for i in range(100):
        die_roll = get_int_1to6(rng)
        is_a_fall = False if get_float_0to1(rng) > 0.005 else True
        if (is_a_fall):
            step_track.append(0)
        else:
            next_pos = step_track[-1] + get_next_move(die_roll, rng)
            if (next_pos < 0):
                step_track.append(step_track[-1])
            else:
                step_track.append(next_pos)

    return step_track


def get_n_walks(n: int, rng: np.random.Generator):
    n_walks = []
    for i in range(n):
        n_walks.append(get_one_walk(rng))

    return n_walks


def main():
    my_rng = get_my_rng()
    n = 500
    all_walks = get_n_walks(n, my_rng)
    end_steps = []

    for i in range(n):
        end_steps.append(all_walks[i][-1])
        # ax.hist(np.array(all_walks[i]))

    count_greater_than_60 = sum(x > 60 for x in end_steps)
    print(end_steps)
    print(count_greater_than_60)

    vals = [12, 61, 75, 40, 90]
    count_gt_60 = sum(x > 60 for x in end_steps)  # -> 3
    print(vals)
    print(count_gt_60)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(end_steps)
    ax.set_title('Random walks')
    ax.set_xlabel('End step')
    plt.show()


# What are the odds that you'll reach 60 steps high on the Empire State Building?
# [55, 43, 76, 99, 61, 98, 51, 66, 77, 51, 7, 4, 36, 48, 56, 27, 50, 0, 46, 27, 69, 67, 60, 84, 76, 79, 58, 40, 15, 8, 74, 79, 4, 41, 0, 84, 51, 13, 81, 75, 1, 26, 67, 0, 67, 4, 68, 53, 107, 58, 104, 44, 74, 61, 19, 71, 45, 62, 80, 66, 76, 18, 27, 26, 5, 81, 7, 0, 52, 74, 0, 91, 89, 47, 80, 88, 52, 68, 80, 82, 76, 49, 98, 55, 55, 85, 62, 82, 66, 28, 75, 64, 1, 68, 86, 57, 75, 66, 8, 111, 22, 75, 3, 69, 50, 74, 38, 63, 24, 75, 23, 80, 36, 78, 53, 53, 39, 47, 21, 9, 65, 52, 83, 43, 99, 63, 92, 38, 78, 74, 32, 0, 31, 9, 76, 4, 86, 18, 42, 91, 68, 10, 26, 60, 4, 72, 27, 57, 87, 87, 78, 20, 86, 63, 9, 98, 64, 88, 72, 85, 106, 44, 55, 70, 51, 68, 66, 98, 72, 69, 84, 91, 57, 47, 33, 88, 61, 8, 52, 104, 88, 65, 75, 52, 80, 63, 45, 85, 12, 61, 50, 81, 42, 93, 34, 102, 52, 43, 104, 39, 45, 51, 64, 66, 83, 72, 7, 74, 80, 14, 81, 73, 65, 83, 76, 81, 66, 98, 22, 75, 57, 62, 76, 17, 7, 101, 90, 56, 68, 82, 32, 70, 52, 98, 77, 25, 109, 49, 93, 25, 75, 83, 96, 71, 11, 60, 59, 37, 72, 1, 97, 45, 101, 75, 67, 26, 63, 26, 31, 111, 76, 96, 73, 81, 91, 33, 54, 51, 30, 16, 62, 97, 70, 86, 24, 73, 67, 85, 6, 89, 38, 65, 86, 55, 42, 51, 80, 58, 87, 56, 18, 81, 4, 41, 91, 50, 72, 45, 70, 56, 3, 23, 82, 12, 12, 101, 82, 67, 88, 57, 50, 3, 108, 63, 68, 56, 60, 55, 58, 81, 64, 42, 68, 80, 29, 59, 72, 55, 62, 92, 9, 115, 62, 64, 78, 80, 70, 103, 42, 85, 80, 17, 82, 1, 82, 81, 60, 95, 84, 39, 86, 77, 63, 41, 56, 67, 75, 81, 20, 68, 2, 77, 7, 27, 89, 64, 53, 106, 82, 113, 18, 19, 70, 72, 80, 43, 81, 92, 60, 71, 28, 77, 90, 17, 16, 68, 63, 63, 91, 69, 56, 110, 73, 70, 93, 18, 89, 22, 35, 81, 2, 67, 78, 85, 13, 25, 67, 66, 62, 61, 85, 57, 24, 81, 108, 15, 56, 21, 40, 58, 87, 90, 24, 64, 39, 15, 3, 5, 87, 78, 54, 50, 107, 74, 2, 9, 37, 82, 11, 68, 78, 35, 60, 71, 103, 74, 87, 63, 78, 86, 52, 13, 62, 28, 76, 83, 22, 67, 77, 26, 55, 76, 34, 72, 79, 23, 32, 62, 51, 63, 91, 40, 99, 64, 30, 93, 49, 12, 40, 1, 41, 102, 71, 24, 77, 10, 71, 26, 71, 28, 53, 47, 90, 64, 6, 46, 85, 30, 112, 55]
# I see 80 values above 60, so 80/500 =>  16%

if __name__ == "__main__":
    main()
