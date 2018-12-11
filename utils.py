'''

implement truncated normal distribution based on the wiki page:

    https://en.wikipedia.org/wiki/Truncated_normal_distribution

'''

import numpy as np
from scipy.special import erf, erfinv


# truely truncated norm
def norm_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * x * x)


def norm_cdf(x):
    return 1 / 2 * (1 + erf(x / np.sqrt(2)))


def norm_icdf(p):  # aka. quantile or ppf
    return np.sqrt(2) * erfinv(2 * p - 1)


def truncnorm_pdf(x, a, b, mu, sigma):
    return norm_pdf((x - mu) / sigma) / (sigma * (
            norm_cdf((b - mu) / sigma) - norm_cdf((a - mu) / sigma)))


def truncnorm_icdf(p, a, b, mu, sigma):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    return norm_icdf(norm_cdf(alpha) + p * (norm_cdf(beta) - norm_cdf(alpha))) * sigma + mu


def truncnorm_gen(a, b, mu, sigma):
    p = np.random.uniform()
    return truncnorm_icdf(p, a, b, mu, sigma)


if __name__ == '__main__':
    N = 100
    vs = [truncnorm_gen(0, 1, 0.1, 1) for _ in range(N)]
    print(min(vs), max(vs), np.mean(vs), np.std(vs))
