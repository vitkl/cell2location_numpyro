import numpyro.distributions as dist
import jax.numpy as jnp


def Gamma(mu=None, sigma=None, alpha=None, beta=None, shape=None):
    r"""
    Function that converts mu/sigma Gamma distribution parametrisation into alpha/beta and
    returns pyro Gamma distribution with an event of a given shape

    :param mu: mean of Gamma distribution
    :param sigma: variance of Gamma distribution
    :param alpha: shape parameter of Gamma distribution (mu ** 2 / sigma ** 2)
    :param beta: rate parameter of Gamma distribution (mu / sigma ** 2)
    :param shape: shape of the event / resulting variable. When None the shape is guessed based on input parameters

    :return: pyro Gamma distribution class
    """
    if alpha is not None and beta is not None:
        pass
    elif mu is not None and sigma is not None:
        alpha = mu ** 2 / sigma ** 2
        beta = mu / sigma ** 2
    else:
        raise ValueError('Define (mu and var) or (alpha and beta).')

    if shape is None:
        alpha = jnp.array(alpha)
        beta = jnp.array(beta)
    else:
        alpha = jnp.ones(shape) * alpha
        beta = jnp.ones(shape) * beta
    return dist.Gamma(alpha, beta)
