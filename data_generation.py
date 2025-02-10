import numpy as np
import scipy.stats as stats

def set_random_seed(seed=42):
    np.random.seed(seed)

# Generate demand using specified probability distribution
def generate_demand(distribution, params, size):
    distributions = {
        'normal': stats.norm.rvs,
        'gamma': stats.gamma.rvs,
        # 'weibull': stats.weibull_min.rvs,
        'poisson': stats.poisson.rvs
    }

    # Validate distribution
    if distribution not in distributions:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # Distribution-specific parameter mapping
    if distribution == 'normal':
        return np.abs(distributions[distribution](loc=params.get('mu', 10),
                                                  scale=params.get('std', 6),
                                                  size=size))
    elif distribution == 'gamma':
        return distributions[distribution](a=params.get('shape', 2),
                                           scale=params.get('scale', 1),
                                           size=size)
    # elif distribution == 'weibull':
    #     return distributions[distribution](c=params.get('shape', 1),
    #                                        scale=params.get('scale', 1),
    #                                        size=size)
    elif distribution == 'poisson':
        return distributions[distribution](mu=params.get('lambda', 2),
                                           size=size)


# Generate lead time using specified probability distribution
def generate_lead_time(distribution, params, size):
    # Use the same distributions as demand generation
    return np.abs(generate_demand(distribution, params, size)).astype(int)
