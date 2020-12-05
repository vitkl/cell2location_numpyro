# This code is adapted from numpyro:
# 1. init_to_mean function for jax
# 2. softplus transformation for Normal Mean Field approximation
#     unconstrained -> softplus -> sigma
# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

import numpy as np

import jax.numpy as jnp
from jax import hessian, lax, random, tree_map
from jax.nn import softplus

from numpyro.infer.autoguide import AutoGuide
from numpyro.infer import init_to_feasible, init_to_median
from numpyro.distributions.transforms import (
    biject_to
)

from numpyro import handlers
from numpyro.distributions.util import periodic_repeat, sum_rightmost

from functools import partial


def sigma_lognormal(locs, scales):
    return (jnp.exp(scales * scales) - 1) * jnp.exp(2 * locs + scales * scales)


def mu_lognormal(locs, scales):
    return jnp.exp(locs + (scales * scales) / 2)


def init_to_mean(site=None):
    """
    Initialize to the prior mean; fallback to median if mean is undefined.
    """
    if site is None:
        return partial(init_to_mean)

    try:
        # Try .mean() method.
        if site['type'] == 'sample' and not site['is_observed'] and not site['fn'].is_discrete:
            value = site["fn"].mean
            # if jnp.isnan(value):
            #    raise ValueError
            if hasattr(site["fn"], "_validate_sample"):
                site["fn"]._validate_sample(value)
            return np.array(value)
    except (NotImplementedError, ValueError):
        # Fall back to a median.
        # This is required for distributions with infinite variance, e.g. Cauchy.
        return init_to_median(site)


class AutoNormal(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Normal distributions
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.
    This should be equivalent to :class: `AutoDiagonalNormal` , but with
    more convenient site names and with better support for mean field ELBO.
    Usage::
        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)
    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`numpyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(self, model, *, prefix="auto", init_loc_fn=init_to_mean, init_scale=0.0,
                 create_plates=None):
        self._init_scale = init_scale
        self._event_dims = {}
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = site["fn"].event_dim + jnp.ndim(self._init_locs[name]) - jnp.ndim(site["value"])
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = self._prototype_frame_full_sizes[frame.name]
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    self._init_locs[name] = periodic_repeat(self._init_locs[name], full_size, dim)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.
        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            init_loc = self._init_locs[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                site_loc = numpyro.param("{}_{}_loc".format(name, self.prefix), init_loc,
                                         event_dim=event_dim)

                # def inv_softplus(x):
                #    return jnp.log(jnp.exp(jnp.abs(x)) - jnp.ones((1, 1)))

                # self._init_scale = inv_softplus(jnp.sqrt(jnp.abs(init_loc)))

                site_scale_unconstrained = numpyro.param("{}_{}_scale".format(name, self.prefix),
                                                         jnp.full(jnp.shape(init_loc), self._init_scale),
                                                         # self._init_scale,
                                                         constraint=constraints.real,
                                                         event_dim=event_dim)
                site_scale = softplus(site_scale_unconstrained)

                site_fn = dist.Normal(site_loc, site_scale).to_event(event_dim)
                if site["fn"].support in [constraints.real, constraints.real_vector]:
                    result[name] = numpyro.sample(name, site_fn)
                else:
                    unconstrained_value = numpyro.sample("{}_unconstrained".format(name), site_fn,
                                                         infer={"is_auxiliary": True})

                    transform = biject_to(site['fn'].support)
                    value = transform(unconstrained_value)
                    log_density = - transform.log_abs_det_jacobian(unconstrained_value, value)
                    log_density = sum_rightmost(log_density,
                                                jnp.ndim(log_density) - jnp.ndim(value) + site["fn"].event_dim)
                    delta_dist = dist.Delta(value, log_density=log_density, event_dim=site["fn"].event_dim)
                    result[name] = numpyro.sample(name, delta_dist)

        return result

    def _constrain(self, latent_samples):
        name = list(latent_samples)[0]
        sample_shape = jnp.shape(latent_samples[name])[
                       :jnp.ndim(latent_samples[name]) - jnp.ndim(self._init_locs[name])]
        if sample_shape:
            flatten_samples = tree_map(lambda x: jnp.reshape(x, (-1,) + jnp.shape(x)[len(sample_shape):]),
                                       latent_samples)
            contrained_samples = lax.map(self._postprocess_fn, flatten_samples)
            return tree_map(lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                            contrained_samples)
        else:
            return self._postprocess_fn(latent_samples)

    def sample_posterior(self, rng_key, params, sample_shape=()):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        scales = {k: params["{}_{}_scale".format(k, self.prefix)] for k in locs}
        with handlers.seed(rng_seed=rng_key):
            latent_samples = {}
            for k in locs:
                latent_samples[k] = numpyro.sample(k, dist.Normal(locs[k], softplus(scales[k])).expand_by(sample_shape))
        return self._constrain(latent_samples)

    def median(self, params):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k, v in self._init_locs.items()}
        return self._constrain(locs)

    def sigma_lognormal(self, params):
        # Change this in the future to also produce estimates for deterministic nodes

        sigma = {k: sigma_lognormal(params["{}_{}_loc".format(k, self.prefix)],
                                    softplus(params["{}_{}_scale".format(k, self.prefix)]))
                 for k, v in self._init_locs.items()}

        return sigma

    def mu_lognormal(self, params):
        # Change this in the future to also produce estimates for deterministic nodes

        mu = {k: mu_lognormal(params["{}_{}_loc".format(k, self.prefix)],
                              softplus(params["{}_{}_scale".format(k, self.prefix)]))
              for k, v in self._init_locs.items()}

        return mu

    def quantiles(self, params, quantiles):
        quantiles = jnp.array(quantiles)[..., None]
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        scales = {k: params["{}_{}_scale".format(k, self.prefix)] for k in locs}
        latent = {k: dist.Normal(locs[k], softplus(scales[k])).icdf(quantiles) for k in locs}
        return self._constrain(latent)
