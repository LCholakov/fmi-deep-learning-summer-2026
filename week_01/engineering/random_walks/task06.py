import numpy as np
import matplotlib.pyplot as plt


def get_my_rng():
    return np.random.default_rng(123)


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
        next_pos = step_track[-1] + get_next_move(die_roll, rng)
        if (next_pos < 0):
            step_track.append(step_track[-1])
        else:
            step_track.append(next_pos)

    return step_track


def get_five_walks(rng: np.random.Generator):
    five_walks = []
    for i in range(5):
        five_walks.append(get_one_walk(rng))

    return five_walks


def main():
    my_rng = get_my_rng()
    all_walks = get_five_walks(my_rng)
    # print (all_walks)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(all_walks[0]))
    ax.plot(np.array(all_walks[1]))
    ax.plot(np.array(all_walks[2]))
    ax.plot(np.array(all_walks[3]))
    ax.plot(np.array(all_walks[4]))

    plt.show()


if __name__ == "__main__":
    main()
