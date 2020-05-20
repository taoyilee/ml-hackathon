import math


def beta_to_mean_std(alpha_q, beta_q):
    inferred_mean = alpha_q / (alpha_q + beta_q)
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)
    return inferred_mean, inferred_std