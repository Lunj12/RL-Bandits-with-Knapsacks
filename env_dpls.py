import numpy as np

# import utils
'''

The RL environment for dynamic pricing with limited supply
'''

# Global Variables
K = 2  # number of products, not arms
d = 1  # number of constrains
n = 6  # maximum number of atoms/arms/actions, must be even
T = 100  # total time step
B = T // 2  # total resources

# Data Processing
mean_valuations = np.random.uniform(low=0, high=1, size=K)
sigma = 0.1


def synthesize_data():
    data = np.random.normal(mean_valuations, [sigma for _ in range(K)], size=(T, K))
    # unconverted truncation
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = min(max(data[i][j], 0), 1)
    return data


valuations = synthesize_data()


def reset_mean_valuation(mean_val=None):
    global mean_valuations
    if mean_val is not None:
        mean_valuations = mean_val
    else:
        mean_valuations = np.random.uniform(low=0, high=1, size=K)


def resample_data():
    global valuations
    valuations = synthesize_data()


# state
class State:
    def __init__(self, t: int, b: int):
        '''

        :param t: time step
        :param b: resources consumption
        '''
        self.t = t
        self.b = b

    def transfer(self, ac: int):
        '''

        :param ac: index of chosen action atom (ia, pr)
        :return: (new State, reward)
        '''
        re, co = reward(self, ac)
        return State(self.t + 1, self.b + co), re, co

    def is_terminated(self):
        return self.t == T or self.b == B


# actions are (product, price) tuples
ARMS = [k for k in range(n)]
PRICES = [(i + 1) / (n / 2 + 1) for i in range(n // 2)]
ACTIONS = [(ac, pr) for ac in ARMS for pr in PRICES]


# reward
def reward(s: State, ac: int):
    '''

    :param s: current state
    :param ac: index of action of (product, price)
    :return: (immediate reward, consumption)
    '''
    # unpack action
    ia, pr = ACTIONS[ac]

    # sample customer values for each product
    vs = valuations[s.t]

    # compare price and valuation for chosen product
    if pr < vs[ia]:
        return pr, 1
    else:
        return 0.0, 0
