from env_dpls import *
import cvxpy

'''
bwkucb is too slow?
'''


def bwkucb(episodes=10):
    r_est = np.zeros((n,))
    c_est = np.zeros((n,))
    nt = np.zeros((n,))
    r_tot, c_tot, st_tot = 0, 0, 0
    policy = [1 / n for _ in range(n)]

    for ep in range(episodes):
        # initial state and actions
        resample_data()
        s = State(0, 0)  # initial state

        while not s.is_terminated():
            if s.t < n:
                ac = s.t
            else:
                conf_r = np.zeros((n,))
                conf_c = np.zeros((n, d))

                C_rad = 1e-2 * np.log(d * s.t)

                for i in range(n):
                    conf_r[i] = np.sqrt(r_est[i] * C_rad / nt[i]) + C_rad / nt[i]
                    conf_c[i] = np.sqrt(c_est[i] * C_rad / nt[i]) + C_rad / nt[i]

                r_ucb = np.array([min(1, r_est[i] + conf_r[i]) for i in range(n)])
                c_lcb = np.array([max(0, c_est[i] - conf_c[i]) for i in range(n)])

                # print(r_est)
                # print(conf_r)

                # Linear Programming to calculate argmax FR(D|miu)
                epsilon = 0.0

                x = cvxpy.Variable(n)  # optimization vector variable for distribution

                obj = cvxpy.Maximize(cvxpy.sum(x * r_ucb))  # define objective function

                cons = [cvxpy.sum(x) == 1]

                for i in range(n):
                    cons = cons + [x[i] >= 0] + [x[i] <= 1]

                for i in range(d):
                    cons = cons + [cvxpy.sum(x * c_lcb) <= B / T * (1 - epsilon)]

                lp_problem = cvxpy.Problem(obj, cons)  # setup the problem
                try:
                    lp_problem.solve()  # solve the problem
                    if x.value is not None:
                        p_opt = list(x.value)  # the optimal variable
                        p_opt = list(map(lambda a: max(min(a, 1), 0), p_opt))
                        policy = [p_opt[k] / sum(p_opt) for k in range(n)]
                except:
                    pass

                ac = np.random.choice(n, p=policy)

            sp, re, co = s.transfer(ac)

            # update info
            nt[ac] += 1
            r_est[ac] = (r_est[ac] * (nt[ac] - 1) + re) / nt[ac]
            c_est[ac] = (c_est[ac] * (nt[ac] - 1) + co) / nt[ac]
            r_tot += re
            c_tot += co

            # move to successor state
            s = sp

        st_tot += s.t

    return r_tot / episodes, c_tot / episodes, st_tot / episodes, policy


def policy_eval(policy=None):
    r_tot, c_tot = 0, 0
    s = State(0, 0)  # initial state

    while not s.is_terminated():
        # pick an (product, price) atom
        ac = np.random.choice(n, p=policy)

        sp, re, co = s.transfer(ac)

        # update info
        r_tot += re
        c_tot += co

        # move to successor state
        s = sp

    return r_tot, c_tot, s.t


if __name__ == '__main__':
    # policy iteration
    avg_info = np.zeros((3,))
    reset_mean_valuation([0.2, 0.4])

    # online policy iteration
    avg_info[0], avg_info[1], avg_info[2], policy = bwkucb(episodes=10)

    print("avg. reward =", avg_info[0])
    print("avg. consumption =", avg_info[1])
    print("avg. steps =", avg_info[2])

    print("")
    print("------------------------------------------")
    print("evaluate optimal policy offline")
    print("optimal policy =", policy)

    # offline policy evaluate
    trails = 1000
    for tr in range(trails):
        resample_data()
        avg_info += policy_eval(policy)

    avg_info /= trails

    print("avg. reward =", avg_info[0])
    print("avg. consumption =", avg_info[1])
    print("avg. steps =", avg_info[2])
