# -*- coding: utf-8 -*-
"""Base pyro model class"""

import os
from collections import defaultdict, Iterable
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax.numpy as jnp
from jax import device_put
from jax import random, jit, lax
import jaxlib
import numpyro as pyro
from numpyro import handlers as poutine  # -
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import Predictive
# from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.autoguide import AutoIAFNormal
# from numpyro.infer.autoguide import AutoDelta
# from numpyro.infer.autoguide import AutoGuideList
import numpyro.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from cell2location.models.base_model import BaseModel
# from cell2location_numpyro.distributions.AutoNormalSoftplus import AutoNormal
# import cell2location_numpyro.distributions.AutoNormalSoftplus
from cell2location_numpyro.distributions.AutoNormal import AutoNormal
from cell2location_numpyro.distributions.AutoNormal import init_to_mean


def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


def flatten_iterable(iterable):
    flattened_list = []
    for i in iterable:
        if isinstance(i, Iterable):
            flattened_list += flatten_iterable(i)
        else:
            flattened_list.append(i)
    return flattened_list


class MiniBatchDataset(Dataset):

    def __init__(self, x_data, extra_data, return_idx=False):
        self.x_data = x_data
        self.extra_data = extra_data
        self.return_idx = return_idx

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.return_idx:
            return self.x_data[idx], {**{'idx': idx}, **{k: v[idx] for k, v in self.extra_data.items()}}
        return self.x_data[idx], {k: v[idx] for k, v in self.extra_data.items()}


# base model class - defining shared methods but not the model itself
class PyroModel(BaseModel):
    r"""Base class for pyro models.
    :param n_fact: number of factors
    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
            self,
            X_data: np.ndarray,
            n_fact: int = 10,
            data_type: str = 'float32',
            n_iter: int = 200000,
            learning_rate=0.001,
            total_grad_norm_constraint=200,
            device: Union['gpu', 'cpu', 'tpu'] = 'gpu',
            verbose=True,
            var_names=None, var_names_read=None,
            obs_names=None, fact_names=None, sample_id=None,
            minibatch_size=None,
            minibatch_seed=42,
            point_estim=[],
            custom_guides={}
    ):

        ############# Initialise parameters ################
        super().__init__(X_data, n_fact,
                         data_type, n_iter,
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)

        self.extra_data = {}
        self.minibatch_size = minibatch_size
        self.minibatch_seed = minibatch_seed
        self.MiniBatchDataset = MiniBatchDataset
        self.device = device
        pyro.set_platform(platform=self.device)
        self.point_estim = point_estim
        self.custom_guides = custom_guides

    def sample_prior(self):
        r""" Take one sample from the prior 
        :return: self.prior_trace dictionary with an element for each parameter of the model. 
        """
        print(".sample_prior() not implemented yet")

    def fit_advi_iterative(self, n: int = 3, method='advi', n_type='restart',
                           n_iter=None, learning_rate=None, progressbar=True,
                           guide_aevb_kwargs={}, random_seed=[5475, 436, 24546, 46],
                           num_particles=1):
        r""" Find posterior using ADVI
        (maximising likehood of the data and minimising KL-divergence of posterior to prior)
        :param n: number of independent initialisations
        :param method: which approximation of the posterior (guide) to use?.
            * ``'advi'`` - Univariate Normal approximation (numpyro.infer.autoguide.AutoNormal)
            * ``'iaf'`` - Inverse Autoregressive Flow approximation
            * ``'custom'`` - Custom guide using conjugate posteriors
        :param guide_aevb_kwargs: arguments for numpyro.infer.autoguide.AutoIAFNormal,
            such as number of hidden layers and the number of nodes in each layer
        :param random_seed: seed needed to use JAX
        :return: self.svi dictionary with svi pyro objects for each n,
            and sefl.hist dictionary storing training history.
        """

        self.random_seed = random_seed
        self.num_particles = num_particles

        # Convert to requested data type
        # self.x_data = self.X_data.astype(self.data_type)

        # initialise parameter store
        self.svi = {}
        self.state = {}
        self.state_param = {}
        self.hist = {}
        self.guide_i = {}
        self.samples = {}
        self.node_samples = {}

        self.n_type = n_type

        if n_iter is None:
            n_iter = self.n_iter

        if learning_rate is None:
            learning_rate = self.learning_rate

        if np.isin(n_type, ['bootstrap']):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ['cv']):
            self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ['init_' + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):

            # initialise Variational distributiion = guide
            if method is 'advi':
                self.guide_i[name] = AutoNormal(self.model,
                                                init_loc_fn=init_to_mean)  # init_to_feasible, init_to_median)

            #elif method is 'iaf':
            #    self.guide_i[name] = AutoIAFNormal(self.model, init_loc_fn=init_to_mean,
            #                                       # init_to_feasible, init_to_median
            #                                       **guide_aevb_kwargs)
            elif method is 'custom':
                self.guide_i[name] = self.guide

            # initialise SVI inference method
            self.svi[name] = SVI(self.model, self.guide_i[name],
                                 # limit the gradient step from becoming too large
                                 optim.ClippedAdam(clip_norm=jnp.array(self.total_grad_norm_constraint),
                                                   **{'step_size': jnp.array(learning_rate)}),
                                 loss=Trace_ELBO(self.num_particles))

            # record ELBO Loss history here
            self.hist[name] = []

            self.set_initial_values()

            # pick dataset depending on the training mode and move to GPU
            if np.isin(n_type, ['cv', 'bootstrap']):
                self.x_data = self.X_data_sample[i].astype(self.data_type)
            else:
                self.x_data = self.X_data.astype(self.data_type)

            # move data to default device
            self.x_data = device_put(self.x_data)

            # train for n_iter
            # self.state[name], self.hist[name] = self.run(name, self.x_data, self.extra_data,
            #                                             random_seed, n_iter, progressbar)

            # self.state[name] = self.step_init(name, self.x_data, self.extra_data, random_seed)
            init_state = self.step_init(name, self.x_data, self.extra_data, self.random_seed[i])
            # init_state = self.svi[name].init(random.PRNGKey(random_seed), x_data=self.x_data)
            # init_state = svi.init(random.PRNGKey(random_seed), x_data=x_data)

            ### fast but does not train
            epochs_iterator = tqdm(range(1))
            for e in epochs_iterator:
                state, losses = lax.scan(lambda state_1, i: self.svi[name].update(state_1,
                                                                                  x_data=self.x_data),
                                         # TODO for minibatch DataLoader goes here
                                         init_state, jnp.arange(n_iter))
                # print(state)
                epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(losses[::-1][0]))
            self.state[name] = state
            self.state_param[name] = self.svi[name].get_params(state).copy()
            self.hist[name] = losses

            ### very slow
            # epochs_iterator = tqdm(range(n_iter))
            # for e in epochs_iterator:
            #    self.state[name], loss = self.step_update(svi=self.svi[name],
            #                                              svi_state=self.state[name],
            #                                              x_data=self.x_data,
            #                                              extra_data=self.extra_data)
            #    self.hist[name].append(loss)
            #    epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(loss))

    def fit_advi_refine(self, n_iter=None):
        r""" Continue training posterior using ADVI
        :param n_iter: number of iterations
        :return: self.svi dictionary with svi pyro objects for each n,
            and sefl.hist dictionary storing training history.
        """

        if n_iter is None:
            n_iter = self.n_iter

        init_names = list(self.state.keys())

        for i, name in enumerate(init_names):

            # pick dataset depending on the training mode and move to GPU
            if np.isin(self.n_type, ['cv', 'bootstrap']):
                self.x_data = self.X_data_sample[i].astype(self.data_type)
            else:
                self.x_data = self.X_data.astype(self.data_type)

            # move data to default device
            self.x_data = device_put(self.x_data)

            ### fast but does not train
            epochs_iterator = tqdm(range(1))
            for e in epochs_iterator:
                state, losses = lax.scan(lambda state_1, i: self.svi[name].update(state_1,
                                                                                  x_data=self.x_data),
                                         # TODO for minibatch DataLoader goes here
                                         self.state[name], jnp.arange(n_iter))
                # print(state)
                epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(losses[::-1][0]))
            self.state[name] = state
            self.state_param[name] = self.svi[name].get_params(state).copy()
            self.hist[name] = list(self.hist[name]) + list(losses)

    def run(self, name, x_data, extra_data,
            random_seed, n_iter, progressbar):
        r"""Function that passes data and extra data to numpyro.run"""

        return self.svi[name].run(rng_key=random.PRNGKey(random_seed),
                                  num_steps=n_iter, progress_bar=progressbar,
                                  x_data=x_data, **extra_data)

    @staticmethod
    def step_update(svi, svi_state, x_data, extra_data):
        r"""Function that passes data and extra data to numpyro.run"""

        def body_fn(svi_state_1):
            return svi.update(svi_state_1, x_data=x_data)  # , **extra_data)

        # return jit(body_fn)(svi_state)
        return jit(body_fn)(svi_state)

    def step_init(self, name, x_data, extra_data, random_seed):

        return self.svi[name].init(random.PRNGKey(random_seed),
                                   x_data=x_data, **extra_data)

    def set_initial_values(self):
        r"""Method for setting initial values (specific to each model or model group)
        :return: nothing
        """
        pass

    def step_train(self, name, x_data, extra_data):

        return self.svi[name].step(x_data)

    def step_eval_loss(self, name, x_data, extra_data):

        return self.svi[name].evaluate_loss(x_data)

    def step_trace(self, name, x_data, extra_data):

        guide_tr = poutine.trace(self.guide_i[name]).get_trace(x_data)
        model_tr = poutine.trace(poutine.replay(self.model,
                                                trace=guide_tr)).get_trace(x_data)
        return guide_tr, model_tr

    def plot_history(self, iter_start: int = 15000, iter_end=-1):
        r""" Plot training history
        :param iter_start: omit initial iterations from the plot
        :param iter_end: omit last iterations from the plot
        """
        for i in self.hist.keys():
            print(plt.plot(np.log10(np.array(self.hist[i])[iter_start:iter_end])));

    @staticmethod
    def predictive(state, guide, num_samples, random_seed):

        # extra_data['x_data'] = x_data

        # return Predictive(model=model, guide=guide, params=extra_data,
        #                  num_samples=num_samples, return_sites=node,
        #                  parallel=True)(rng_key=random.PRNGKey(random_seed),
        #                                 **extra_data)
        return guide.sample_posterior(random.PRNGKey(random_seed), state,
                                      sample_shape=(num_samples,)).copy()

    @staticmethod
    def predictive_quantile(params, guide, quantiles):

        return guide.quantiles(params, quantiles)

    @staticmethod
    def predictive_sigma_lognormal(params, guide):
        sigma = guide.sigma_lognormal(params=params)
        return sigma

    @staticmethod
    def predictive_mu_lognormal(params, guide):
        mu = guide.mu_lognormal(params=params)
        return mu

    def sample_node1(self, node, init, batch_size: int = 50, random_seed=65756):

        post_samples = self.predictive(state=self.state_param[init],  # self.svi[init].get_params(self.state[init]),
                                       guide=self.guide_i[init],
                                       num_samples=batch_size, random_seed=random_seed)
        # self.predictive(model=self.model, guide=self.guide_i[init],
        #                               x_data=self.x_data, extra_data=self.extra_data,
        #                               num_samples=batch_size, node=[node],
        #                               random_seed=random_seed)
        post_samples_np = {k: np.array(v)
                           for k, v in post_samples.items() if k == node}
        del post_samples
        import gc
        gc.collect()

        return post_samples_np[node]

    def sample_node(self, node, init, n_sampl_iter,
                    batch_size: int = 1, suff='', random_seed=65756):

        if batch_size == 1:
            self.samples[node + suff][init] = np.array(self.predictive_quantile(params=self.state_param[init],
                                                                                guide=self.guide_i[init],
                                                                                quantiles=0.5)[node])
        else:
            # sample first batch
            self.samples[node + suff][init] = self.sample_node1(node, init, batch_size=batch_size,
                                                                random_seed=random_seed)

            for it in tqdm(range(n_sampl_iter - 1)):
                # sample remaining batches
                post_node = self.sample_node1(node, init, batch_size=batch_size,
                                              random_seed=random_seed)

                # concatenate batches
                self.samples[node + suff][init] = np.concatenate((self.samples[node + suff][init], post_node), axis=0)

            # compute mean across samples
            self.samples[node + suff][init] = np.array(self.samples[node + suff][init].mean(0))

    def sample_all1(self, init='init_1', batch_size: int = 50, random_seed=65756):

        # nodes = self.state[init].keys
        # nodes = nodes[nodes != "data_target"]

        post_samples = self.predictive(state=self.state_param[init],  # self.svi[init].get_params(self.state[init]),
                                       guide=self.guide_i[init],
                                       num_samples=batch_size, random_seed=random_seed)
        # self.predictive(model=self.model, guide=self.guide_i[init],
        #                               x_data=self.x_data, extra_data=self.extra_data,
        #                               num_samples=batch_size, node=[node],
        #                               random_seed=random_seed)

        # post_samples = self.predictive(model=self.model, guide=self.guide_i[init],
        #                               x_data=self.x_data, extra_data=self.extra_data,
        #                               num_samples=batch_size,
        #                               random_seed=random_seed, node=self.node_list)
        post_samples_np = {k: np.array(v)
                           for k, v in post_samples.items() if k != "data_target"}
        del post_samples
        import gc
        gc.collect()

        return post_samples_np

    def sample_all(self, n_sampl_iter, init='init_1', batch_size: int = 50, random_seed=65756):

        # sample first batch
        self.samples['post_samples'] = self.sample_all1(init, batch_size=batch_size,
                                                        random_seed=random_seed)

        for it in tqdm(range(n_sampl_iter - 1)):
            # sample remaining batches
            post_samples = self.sample_all1(init, batch_size=batch_size)

            # concatenate batches
            self.samples['post_samples'] = {k: np.concatenate((self.samples['post_samples'][k],
                                                               post_samples[k]), axis=0)
                                            for k in post_samples.keys()}

    def b_evaluate_stability(self, node, fact_filt=None, quantile=True, n_samples: int = 1, batch_size: int = 10,
                             align=True, transpose=True, random_seed=65756):
        r""" Evaluate stability of posterior samples between training initialisations
        (takes samples and correlates the values of factors between training initialisations)
        :param node: which pymc3 node to sample? Factors should be in columns.
        :param n_samples: the number of samples.
        :param batch_size: generate samples in batches of size `batch_size`. Necessary for the computation to fit in the GPU memory
        :return: self.samples[node_name+_stab] dictionary with an element for each training initialisation.
        """

        if quantile:
            n_samples = 1
        self.n_samples = n_samples
        self.n_sampl_iter = int(np.ceil(n_samples / batch_size))
        if n_samples == 1:
            self.n_sampl_batch = 1
        else:
            self.n_sampl_batch = batch_size

        self.samples[node + '_stab'] = {}

        for i in self.guide_i.keys():
            self.sample_node(node, i, self.n_sampl_iter,
                             batch_size=self.n_sampl_batch, suff='_stab',
                             random_seed=random_seed)

        # plot correlations of posterior mean between training initialisations
        for i in range(len(self.samples[node + '_stab'].keys()) - 1):
            x = self.samples[node + '_stab']['init_' + str(1)]
            y = self.samples[node + '_stab']['init_' + str(i + 2)]
            if transpose:
                x = x.T
                y = y.T

            if fact_filt is not None:
                x = x[:, fact_filt]
                y = y[:, fact_filt]
            print(self.align_plot_stability(x, y,
                                            str(1), str(i + 2), align=align))

    def sample_posterior(self, node='all', quantile=True,
                         n_samples: int = 100, batch_size: int = 10,
                         save_samples=False,
                         mean_field_slot='init_1', random_seed=65756):
        r""" Sample posterior distribution of parameters - either all or single parameter
        :param node: pyro parameter to sample (e.g. default "all", self.spot_factors)
        :param n_samples: number of posterior samples to generate (1000 is recommended, reduce if you get GPU memory error)
        :param save_samples: save samples in addition to sample mean, 5% quantile, SD.
        :param return_samples: return summarised samples in addition to saving them in `self.samples`
        :param mean_field_slot: string, which mean_field slot to sample? 'init_1' by default
        :return: dictionary of dictionaries (mean, 5% quantile, SD, optionally all samples) with numpy arrays for each parameter.
        Optional dictionary of all samples contains parameters as numpy arrays of shape ``(n_samples, ...)``
        """

        self.n_samples = n_samples
        self.n_sampl_iter = int(np.ceil(n_samples / batch_size))
        self.n_sampl_batch = batch_size

        if (node == 'all'):
            # Sample all parameters - might use a lot of GPU memory
            if quantile:
                self.samples['post_sample_means'] = self.predictive_quantile(params=self.state_param[mean_field_slot],
                                                                             guide=self.guide_i[mean_field_slot],
                                                                             quantiles=0.5)
                param_names = list(self.samples['post_sample_means'].keys())
                self.samples['post_sample_means'] = {v: np.array(self.samples['post_sample_means'][v])
                                                     for v in param_names}

                self.samples['post_sample_q05'] = self.predictive_quantile(params=self.state_param[mean_field_slot],
                                                                           guide=self.guide_i[mean_field_slot],
                                                                           quantiles=0.05)
                self.samples['post_sample_q05'] = {v: np.array(self.samples['post_sample_q05'][v])
                                                   for v in param_names}

                self.samples['post_sample_q95'] = self.predictive_quantile(params=self.state_param[mean_field_slot],
                                                                           guide=self.guide_i[mean_field_slot],
                                                                           quantiles=0.95)
                self.samples['post_sample_q95'] = {v: np.array(self.samples['post_sample_q95'][v])
                                                   for v in param_names}

                self.samples['post_sample_sds'] = self.predictive_sigma_lognormal(
                    params=self.state_param[mean_field_slot],
                    guide=self.guide_i[mean_field_slot])
                param_names = list(self.samples['post_sample_sds'].keys())
                self.samples['post_sample_sds'] = {v: np.array(self.samples['post_sample_sds'][v])
                                                   for v in param_names}

                self.samples['post_sample_mu_lognormal'] = self.predictive_mu_lognormal(
                    params=self.state_param[mean_field_slot],
                    guide=self.guide_i[mean_field_slot])
                param_names = list(self.samples['post_sample_mu_lognormal'].keys())
                self.samples['post_sample_mu_lognormal'] = {v: np.array(self.samples['post_sample_mu_lognormal'][v])
                                                            for v in param_names}

            else:

                self.sample_all(self.n_sampl_iter, init=mean_field_slot, batch_size=self.n_sampl_batch,
                                random_seed=random_seed)

                param_names = list(self.samples['post_samples'].keys())

                self.samples['post_sample_means'] = {v: np.array(self.samples['post_samples'][v].mean(axis=0))
                                                     for v in param_names}
                self.samples['post_sample_q05'] = {
                    v: np.array(np.quantile(self.samples['post_samples'][v], 0.05, axis=0))
                    for v in param_names}
                self.samples['post_sample_q95'] = {
                    v: np.array(np.quantile(self.samples['post_samples'][v], 0.95, axis=0))
                    for v in param_names}
                self.samples['post_sample_sds'] = {v: np.array(self.samples['post_samples'][v].std(axis=0))
                                                   for v in param_names}

                if not save_samples:
                    self.samples['post_samples'] = None

        else:
            self.sample_node(node, mean_field_slot, self.n_sampl_iter,
                             batch_size=self.n_sampl_batch, suff='',
                             random_seed=random_seed)

        return self.samples

    def save_checkpoint(self, n, prefix=''):
        r""" Save pyro parameter store (current status of Variational parameters) to disk
        :param n: epoch number
        :param prefix: filename prefix (e.g. init number)
        """

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        for i in self.state.keys():
            filename = f'{self.checkpoint_dir}/{prefix}_{i}_{n}.ckp'
            self.state[i].save(filename)

    def load_checkpoint(self, filename):
        r""" Load pyro parameter store (current status of Variational parameters) from disk
        :param filename: checkpoint filename
        """

        if filename in os.listdir(self.checkpoint_dir):
            pyro.get_param_store().save(filename)
        else:
            checkpoints = os.listdir(self.checkpoint_dir)
            checkpoints = '\n'.join(checkpoints)
            checkpoint_dir_abspath = os.path.abspath(self.checkpoint_dir)
            raise ValueError(f'No such filename in {checkpoint_dir_abspath}, available filenames : \n'
                             f'{checkpoints}')

    def fit_nuts(self, n_samples: int = 1000, warmup_steps: int = 1000, save_samples=False):

        self.samples = {}
        self.n_samples = n_samples

        # create sampler and run MCMC
        self.nuts_kernel = NUTS(self.model, jit_compile=True)
        self.mcmc = MCMC(self.nuts_kernel, num_samples=n_samples,
                         warmup_steps=warmup_steps)
        self.mcmc.run(self.x_data)

        post_samples = {k: v.detach().cpu().numpy() for k, v in self.mcmc.get_samples().items()}

        # summarise samples
        self.samples['post_sample_means'] = {v: post_samples[v].mean(axis=0) for v in post_samples.varnames}
        self.samples['post_sample_q05'] = {v: np.quantile(post_samples[v], 0.05, axis=0) for v in post_samples.varnames}
        self.samples['post_sample_q95'] = {v: np.quantile(post_samples[v], 0.95, axis=0) for v in post_samples.varnames}
        self.samples['post_sample_sds'] = {v: post_samples[v].std(axis=0) for v in post_samples.varnames}

        if (save_samples):
            self.samples['post_samples']
