from env_dpls import *


# e-soft policy (e.g. epsilon-greedy) for all Q(s,a)
def egreedy_policy(Q, epsilon):
    policy = np.zeros((T + 1, B + 1, n), dtype=float)
    for tt in range(Q.shape[0]):
        for bb in range(Q.shape[1]):
            i_opt = np.argmax(Q[tt, bb, :])
            for ii in range(n):
                if ii == i_opt:
                    policy[tt, bb, ii] = 1 - epsilon + epsilon / n
                else:
                    policy[tt, bb, ii] = epsilon / n
    return policy


# softmax policy (e.g. epsilon-greedy) for all Q(s,a)
def softmax_policy(Q, episode):  # e-soft policy (e.g. epsilon-greedy)
    gamma = 1e-2
    tau0 = 298
    tau = tau0 * np.exp(-gamma * episode)

    policy = np.zeros((T + 1, B + 1, n), dtype=float)
    for tt in range(Q.shape[0]):
        for bb in range(Q.shape[1]):
            policy[tt, bb, :] = np.exp(Q[tt, bb, :] / tau) / np.sum(np.exp(Q[tt, bb, :] / tau))
    return policy


def monte_carlo(episodes=10, epsilon=0.1, gamma=0.9):
    Q = np.random.rand(T + 1, B + 1, n)  # t in [0, T], b in [0, B+1], ac in [0, n-1]
    policy = egreedy_policy(Q, epsilon)  # initial policy

    Returns_sum = np.zeros((T + 1, B + 1, n), dtype=float)
    Returns_count = np.zeros((T + 1, B + 1, n), dtype=int)

    r_tot, c_tot, st_tot = 0, 0, 0

    # generate an entire episode using one state-based policy
    for ep in range(episodes):
        resample_data()
        episode_path = []
        # initial state, actions, and returns
        s = State(0, 0)

        while not s.is_terminated():
            ac = np.random.choice(n, p=policy[s.t, s.b, :])
            # take an action
            sp, re, co = s.transfer(ac)

            # update info after each step
            episode_path.append((s.t, s.b, ac, re))
            r_tot += re
            c_tot += co

            # move to successor state
            s = sp

        st_tot += s.t

        # update Q after each episode: total discounted reward following first occurrence of (s,a)
        visited = set()
        for (i_first, x) in enumerate(episode_path):
            t, b, ac, re = x
            if (t, b, ac) in visited:  # skip non-first occurrence
                continue

            visited.add((t, b, ac))
            R = sum([x[3] * (gamma ** i) for (i, x) in enumerate(episode_path[i_first:])])
            Returns_sum[t, b, ac] += R
            Returns_count[t, b, ac] += 1
            Q[t, b, ac] = Returns_sum[t, b, ac] / Returns_count[t, b, ac]

        # update policy after each episode:
        policy = egreedy_policy(Q, epsilon)
        # policy = softmax_policy(Q, ep)

    return r_tot / episodes, c_tot / episodes, st_tot / episodes


if __name__ == '__main__':
    avg_info = np.zeros((3,))
    reset_mean_valuation([0.1, 0.4])

    # quick view
    # avg_info = monte_carlo(episodes=1000, epsilon=0.1, gamma=0.75)
    # print("avg. reward =", avg_info[0])
    # print("avg. consumption =", avg_info[1])
    # print("avg. steps =", avg_info[2])

    # for convergence test
    test_data = []
    for ep in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]:
        avg_info = monte_carlo(episodes=ep, epsilon=0.1, gamma=0.75)
        print("total episodes =", ep, "avg. reward =", avg_info[0])
        test_data.append(avg_info[0])
