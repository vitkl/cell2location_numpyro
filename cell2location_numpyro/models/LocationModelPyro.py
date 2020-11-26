r"""Location model decomposes the expression of genes across locations into a set of reference regulatory programmes,
    it is identical to LocationModelLinearDependentWPyro but does not account for linear dependencies in abundance of programs
    across locations with similar cell composition."""
from typing import Union

import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
import torch
from numpyro import handlers as poutine
import jax.numpy as jnp
from numpyro.infer import Predictive
from jax import random

from cell2location_numpyro.models.pyro_loc_model import PyroLocModel
from cell2location_numpyro.distributions.Gamma import Gamma
from numpyro.distributions import GammaPoisson

def rand_tensor(shape, mean, sigma):
    r""" Helper for initializing variational parameters
    """
    return mean * torch.ones(shape) + sigma * torch.randn(shape)


########-------- defining the model itself - pyro -------- ########
class LocationModelPyro(PyroLocModel):
    r"""Provided here as a 'base' model for completeness.

    Parameters
    ----------
    cell_state_mat :
        Pandas data frame with gene programmes - genes in rows, cell types / factors in columns
    X_data :
        Numpy array of gene expression (cols) in spatial locations (rows)
    n_iter :
        number of training iterations
    learning_rate, data_type, total_grad_norm_constraint, ...:
        See parent class BaseModel for details.
    gene_level_prior :
        see the description for LocationModelLinearDependentWPyro
    gene_level_var_prior :
        see the description for LocationModelLinearDependentWPyro
    cell_number_prior :
        see the description for LocationModelLinearDependentWPyro, this model does not have **combs_per_spot**
        parameter.
    cell_number_var_prior :
        see the description for LocationModelLinearDependentWPyro, this model does not have
        **combs_mean_var_ratio** parameter.
    phi_hyp_prior :
        see the description for LocationModelLinearDependentWPyro

    Returns
    -------

    """

    def __init__(
            self,
            cell_state_mat: np.ndarray,
            X_data: np.ndarray,
            data_type: str = 'float32',
            n_iter=30000,
            learning_rate=0.005,
            total_grad_norm_constraint=200,
            device: Union['gpu', 'cpu', 'tpu'] = 'gpu',
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            gene_level_prior={'mean': 1 / 2, 'sd': 1 / 4},
            gene_level_var_prior={'mean_var_ratio': 1},
            cell_number_prior={'cells_per_spot': 8,
                               'factors_per_spot': 7},
            cell_number_var_prior={'cells_mean_var_ratio': 1,
                                   'factors_mean_var_ratio': 1},
            phi_hyp_prior={'mean': 3, 'sd': 1},
            initialise_at_prior=False,
            minibatch_size=None,
            minibatch_seed=42
    ):

        ############# Initialise parameters ################
        super().__init__(cell_state_mat=cell_state_mat, X_data=X_data,
                         data_type=data_type, n_iter=n_iter,
                         learning_rate=learning_rate, total_grad_norm_constraint=total_grad_norm_constraint,
                         device=device, verbose=verbose, var_names=var_names, var_names_read=var_names_read,
                         obs_names=obs_names, fact_names=fact_names, sample_id=sample_id,
                         minibatch_size=minibatch_size, minibatch_seed=minibatch_seed)

        self.gene_level_prior = gene_level_prior
        self.cell_number_prior = cell_number_prior
        self.phi_hyp_prior = phi_hyp_prior
        self.initialise_at_prior = initialise_at_prior

        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        self.cell_number_prior = cell_number_prior

        for k in gene_level_var_prior.keys():
            gene_level_prior[k] = gene_level_var_prior[k]
        self.gene_level_prior = gene_level_prior

        ## define parameters used to initialize variational parameters##
        self.gl_shape = self.gene_level_prior['mean'] ** 2 / self.gene_level_prior['sd'] ** 2
        self.gl_rate = self.gene_level_prior['mean'] / self.gene_level_prior['sd'] ** 2
        self.gl_shape_var = self.gl_shape / self.gene_level_prior['mean_var_ratio']
        self.gl_rate_var = self.gl_rate / self.gene_level_prior['mean_var_ratio']
        self.gl_mean = self.gene_level_prior['mean']
        self.gl_sd = self.gene_level_prior['sd']

    ############# Define the model ################
    def model(self, x_data, idx=None):

        # =====================Gene expression level scaling======================= #
        # Explains difference in expression between genes and
        # how it differs in single cell and spatial technology
        # compute hyperparameters from mean and sd

        shape = self.gene_level_prior['mean'] ** 2 / self.gene_level_prior['sd'] ** 2
        rate = self.gene_level_prior['mean'] / self.gene_level_prior['sd'] ** 2
        shape_var = shape / self.gene_level_prior['mean_var_ratio']
        rate_var = rate / self.gene_level_prior['mean_var_ratio']

        n_g_prior = np.array(self.gene_level_prior['mean']).shape
        if len(n_g_prior) == 0:
            n_g_prior = 1
        else:
            n_g_prior = self.n_var

        self.gene_level_alpha_hyp = pyro.sample('gene_level_alpha_hyp',
                                                Gamma(mu=shape,
                                                      sigma=np.sqrt(shape_var),
                                                      shape=(n_g_prior, 1)))

        self.gene_level_beta_hyp = pyro.sample('gene_level_beta_hyp',
                                               Gamma(mu=rate,
                                                     sigma=np.sqrt(rate_var),
                                                     shape=(n_g_prior, 1)))

        self.gene_level = pyro.sample('gene_level',
                                      Gamma(alpha=self.gene_level_alpha_hyp,
                                            beta=self.gene_level_beta_hyp,
                                            shape=(self.n_var, 1)))

        # scale cell state factors by gene_level
        self.gene_factors = pyro.deterministic('gene_factors', self.cell_state)

        # =====================Spot factors======================= #
        # prior on spot factors reflects the number of cells, fraction of their cytoplasm captured,
        # times heterogeniety in the total number of mRNA between individual cells with each cell type
        cps_shape = self.cell_number_prior['cells_per_spot'] ** 2 \
                    / (self.cell_number_prior['cells_per_spot'] / self.cell_number_prior['cells_mean_var_ratio'])
        cps_rate = self.cell_number_prior['cells_per_spot'] \
                   / (self.cell_number_prior['cells_per_spot'] / self.cell_number_prior['cells_mean_var_ratio'])
        self.cells_per_spot = pyro.sample('cells_per_spot',
                                          dist.Gamma(jnp.ones([self.n_obs, 1]) * jnp.array(cps_shape),
                                                     jnp.ones([self.n_obs, 1]) * jnp.array(cps_rate)))

        fps_shape = self.cell_number_prior['factors_per_spot'] ** 2 \
                    / (self.cell_number_prior['factors_per_spot'] / self.cell_number_prior['factors_mean_var_ratio'])
        fps_rate = self.cell_number_prior['factors_per_spot'] \
                   / (self.cell_number_prior['factors_per_spot'] / self.cell_number_prior['factors_mean_var_ratio'])
        self.factors_per_spot = pyro.sample('factors_per_spot',
                                            dist.Gamma(jnp.ones([self.n_obs, 1]) * jnp.array(fps_shape),
                                                       jnp.ones([self.n_obs, 1]) * jnp.array(fps_rate)))

        shape = self.factors_per_spot / jnp.array(np.array(self.n_fact).reshape((1, 1)))
        rate = jnp.ones([1, 1]) / self.cells_per_spot * self.factors_per_spot
        self.spot_factors = pyro.sample('spot_factors',
                                        dist.Gamma(jnp.dot(shape, jnp.ones([1, self.n_fact])),
                                                   jnp.dot(rate, jnp.ones([1, self.n_fact]))))

        # =====================Spot-specific additive component======================= #
        # molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed between all genes not just expressed genes
        self.spot_add_hyp = pyro.sample('spot_add_hyp',
                                        dist.Gamma(jnp.ones([2, 1]) * jnp.array(1.),
                                                   jnp.ones([2, 1]) * jnp.array(0.1)))
        self.spot_add = pyro.sample('spot_add',
                                    dist.Gamma(jnp.ones([self.n_obs, 1]) * self.spot_add_hyp[0, 0],
                                               jnp.ones([self.n_obs, 1]) * self.spot_add_hyp[1, 0]))

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by cell state signatures
        # these counts are distributed equally between all spots (e.g. background, free-floating RNA)
        self.gene_add_hyp = pyro.sample('gene_add_hyp',
                                        dist.Gamma(jnp.ones([2, 1]) * jnp.array(1.),
                                                   jnp.ones([2, 1]) * jnp.array(1.)))
        self.gene_add = pyro.sample('gene_add',
                                    dist.Gamma(jnp.ones([self.n_var, 1]) * self.gene_add_hyp[0, 0],
                                               jnp.ones([self.n_var, 1]) * self.gene_add_hyp[1, 0]))

        # =====================Gene-specific overdispersion ======================= #
        self.phi_hyp = pyro.sample('phi_hyp',
                                   dist.Gamma(jnp.ones([1, 1]) * jnp.array(self.phi_hyp_prior['mean']),
                                              jnp.ones([1, 1]) * jnp.array(self.phi_hyp_prior['sd'])))
        self.gene_E = pyro.sample('gene_E', dist.Exponential(jnp.ones([self.n_var, 1]) * self.phi_hyp[0, 0]))

        # =====================Expected expression ======================= #
        # expected expression
        self.mu_biol = jnp.dot(self.spot_factors[idx], self.gene_factors.T) * self.gene_level.T \
                       + self.gene_add.T + self.spot_add[idx]
        self.theta = jnp.ones([1, 1]) / (self.gene_E.T * self.gene_E.T)

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        self.data_target = pyro.sample('data_target',
                                       GammaPoisson(rate=self.theta / self.mu_biol,
                                                    concentration=self.theta),
                                       obs=x_data)

        # =====================Compute nUMI from each factor in spots  ======================= #
        nUMI = (self.spot_factors * (self.gene_factors * self.gene_level).sum(0))
        self.nUMI_factors = pyro.deterministic('nUMI_factors', nUMI)

    def compute_expected(self):
        r"""Compute expected expression of each gene in each spot (Poisson mu). Useful for evaluating how well
        the model learned expression pattern of all genes in the data.

        """

        # compute the poisson rate
        self.mu = (np.dot(self.samples['post_sample_means']['spot_factors'],
                          self.samples['post_sample_means']['gene_factors'].T)
                   * self.samples['post_sample_means']['gene_level'].T
                   + self.samples['post_sample_means']['gene_add'].T
                   + self.samples['post_sample_means']['spot_add'])

    def run(self, name, x_data, extra_data,
            random_seed, n_iter, progressbar):
        r"""Function that passes data and extra data to numpyro.run"""

        idx = np.arange(x_data.shape[0])
        return self.svi[name].run(rng_key=random.PRNGKey(random_seed),
                                  num_steps=n_iter, progress_bar=progressbar,
                                  x_data=x_data, idx=idx, **extra_data)

    def predictive(self, model, guide, x_data, extra_data, num_samples, node):

        idx = extra_data.get('idx')
        if idx is None:
            extra_data['idx'] = np.arange(x_data.shape[0])
        extra_data['x_data'] = x_data

        return Predictive(model=model, guide=guide, params=extra_data,
                          num_samples=num_samples, return_sites=node,
                          parallel=True)

    def step_train(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return self.svi[name].step(x_data, idx)

    def step_eval_loss(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return self.svi[name].evaluate_loss(x_data, idx)

    def step_predictive(self, predictive, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        return predictive(x_data, idx)

    def step_trace(self, name, x_data, extra_data):
        idx = extra_data.get('idx')
        if idx is None:
            idx = torch.LongTensor(np.arange(x_data.shape[0]))
        guide_tr = poutine.trace(self.guide_i[name]).get_trace(x_data, idx)
        model_tr = poutine.trace(poutine.replay(self.model,
                                                trace=guide_tr)).get_trace(x_data, idx)
        return guide_tr, model_tr
