from env_dpls import *


def qlearning(episodes=10, epsilon=0.1, lr=1e-5, gamma=0.9):
    Q = np.random.rand(T + 1, B + 1, n)  # t in [0, T], b in [0, B], ac in [0, n-1]
    r_tot, c_tot, st_tot = 0, 0, 0

    for ep in range(episodes):
        resample_data()
        # initial state
        s = State(0, 0)

        while not s.is_terminated():
            # q-learning off-policy: after an action, choose next action base on updated max Q
            if np.random.rand() <= epsilon:
                ac = np.random.choice(n)
            else:
                ac = np.argmax(Q[s.t, s.b, :])

            # take an action
            sp, re, co = s.transfer(ac)

            # update Q and other info
            Q[s.t, s.b, ac] = Q[s.t, s.b, ac] + lr * (re + gamma * max(Q[sp.t, sp.b, :]) - Q[s.t, s.b, ac])

            r_tot += re
            c_tot += co

            # move to successor state
            s = sp

        st_tot += s.t

    return r_tot / episodes, c_tot / episodes, st_tot / episodes


if __name__ == '__main__':
    avg_info = np.zeros((3,))
    reset_mean_valuation([0.2, 0.4])

    # quick view
    # avg_info = qlearning(episodes=3000, epsilon=0.01, lr=0.9, gamma=0.75)
    # print("avg. reward =", avg_info[0])
    # print("avg. consumption =", avg_info[1])
    # print("avg. steps =", avg_info[2])

    # for convergence test
    test_data = []
    for ep in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]:
        avg_info = qlearning(episodes=ep, epsilon=0.1, lr=0.9, gamma=0.75)
        print("total episodes =", ep, "avg. reward =", avg_info[0])
        test_data.append(avg_info[0])
