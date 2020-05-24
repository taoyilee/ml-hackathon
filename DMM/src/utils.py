def beta_to_mean_std(alpha_q, beta_q):
    inferred_mean = alpha_q / (alpha_q + beta_q)
    inferred_std = (alpha_q * beta_q) / (((alpha_q + beta_q) ** 2) * (alpha_q + beta_q + 1))
    return inferred_mean, inferred_std
