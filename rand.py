from env_dpls import *


def rand():
    r_tot, c_tot = 0, 0
    s = State(0, 0)  # initial state

    while not s.is_terminated():
        # pick an (product, price) atom
        ac = np.random.choice(n)
        sp, re, co = s.transfer(ac)

        # update info
        r_tot += re
        c_tot += co

        # move to successor state
        s = sp

    return r_tot, c_tot, s.t


if __name__ == '__main__':
    trails = 10000
    avg_info = np.zeros((3,))
    reset_mean_valuation([0.2, 0.4])

    for tr in range(trails):
        resample_data()
        avg_info += rand()

    avg_info /= trails

    print("avg. reward =", avg_info[0])
    print("avg. consumption =", avg_info[1])
    print("avg. steps =", avg_info[2])
