from env_dpls import *


def egreedy(episodes=10, epsilon=0.01):
    r_tot, c_tot, st_tot = 0, 0, 0
    r_est = np.zeros((n,))
    nt = np.zeros((n,))

    for ep in range(episodes):
        # initial state and actions
        resample_data()
        s = State(0, 0)  # initial state
        while not s.is_terminated():
            # pick an (product, price) atom
            if np.random.rand() <= epsilon:
                ac = np.random.choice(n)
            else:
                ac = np.argmax(r_est)

            sp, re, co = s.transfer(ac)

            # update info
            nt[ac] += 1
            r_tot += re
            c_tot += co
            r_est[ac] = (r_est[ac] * (nt[ac] - 1) + re) / nt[ac]

            # move to successor state
            s = sp

        st_tot += s.t

    return r_tot / episodes, c_tot / episodes, st_tot / episodes


if __name__ == '__main__':
    avg_info = np.zeros((3,))
    reset_mean_valuation([0.2, 0.4])

    # quick view
    # avg_info = egreedy(episodes=1000, epsilon=0.1)
    # print("avg. reward =", avg_info[0])
    # print("avg. consumption =", avg_info[1])
    # print("avg. steps =", avg_info[2])

    # for convergence test
    test_data = []
    for ep in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]:
        avg_info = egreedy(episodes=ep, epsilon=0.1)
        print("total episodes =", ep, "avg. reward =", avg_info[0])
        test_data.append(avg_info[0])
