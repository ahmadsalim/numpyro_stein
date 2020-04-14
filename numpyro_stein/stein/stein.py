from functools import namedtuple
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import transform_fn, log_density
from numpyro_stein.stein.autoguides import AutoDelta
from numpyro_stein.stein.kernels import SteinKernel
from numpyro_stein.util import ravel_pytree
from typing import Callable

import jax
import jax.random
import jax.numpy as np
from jax.tree_util import tree_map

# TODO, next steps.
# * Test multivariate RBF kernel support
# * Optimize running the implementation with compiled loops (look at fori_collect and how it is used in NumPyro)
# * Implement IMQ kernel like in Pyro (Measuring Sample Quality)
# * Implement linear and random kernel
# * Implement Matrix valued kernels (For second-order stuff)
# * Implement Stein Point MCMC updates
# * Integrate projected SVN ideas in matrix valued kernels/inference


SVGDState = namedtuple('SVGDState', ['optim_state', 'rng_key'])

# Lots of code based on SVI interface and commonalities should be refactored
class SVGD(object):
    """
    Stein Variational Gradient Descent for Non-parametric Inference.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param kernel_fn: Function that produces a logarithm of the statistical kernel to use with Stein inference
    :param num_stein_particles: number of particles for Stein inference.
        (More particles capture more of the posterior distribution)
    :param num_loss_particles: number of particles to evaluate the loss.
        (More loss particles reduce variance of loss estimates for each Stein particle)
    :param loss_temperature: scaling of loss factor
    :param repulsion_temperature: scaling of repulsive forces (Non-linear SVGD)
    :param classic_guide_param_fn: predicate on names of parameters in guide which should be optimized classically without Stein
            (E.g., parameters for large normal networks or other transformation)
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """
    def __init__(self, model, guide, optim, loss, kernel_fn: SteinKernel,
                 num_stein_particles: int=10, num_loss_particles: int=2,
                 loss_temperature: float=1.0, repulsion_temperature: float=1.0, 
                 classic_guide_params_fn: Callable[[str], bool]=lambda name: False, **static_kwargs):
        assert isinstance(guide, AutoDelta) # Only AutoDelta guide supported for now
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.kernel_fn = kernel_fn
        self.static_kwargs = static_kwargs
        self.num_stein_particles = num_stein_particles
        self.num_loss_particles = num_loss_particles
        self.loss_temperature = loss_temperature
        self.repulsion_temperature = repulsion_temperature
        self.classic_guide_params_fn = classic_guide_params_fn
        self.guide_param_names = None
        self.constrain_fn = None

    def _kernel_grad(self, kernel, x, y):
        if self.kernel_fn.mode() == 'norm':
            return jax.grad(lambda x: kernel(x, y))(x)
        elif self.kernel_fn.mode() == 'vector':
            return jax.vmap(lambda i: jax.grad(lambda xi: kernel(xi, y[i]))(x[i]))(enumerate(x.shape))
        else:
            assert False, 'Non-supported kernel model'

    def _svgd_loss_and_grads(self, rng_key, unconstr_params, *args, **kwargs):
        # 0. Separate model and guide parameters, since only guide parameters are updated using Stein
        classic_uparams = {p: v for p, v in unconstr_params.items() if p not in self.guide_param_names or self.classic_guide_params_fn(p)}
        stein_uparams = {p: v for p, v in unconstr_params.items() if p not in classic_uparams}
        # 1. Collect each guide parameter into monolithic particles that capture correlations between parameter values across each individual particle
        stein_particles, unravel_pytree, unravel_pytree_batched = ravel_pytree(stein_uparams, batch_dims=1)

        # 2. Calculate loss and gradients for each parameter (broadcasting by num_loss_particles for increased variance reduction)
        def scaled_loss(rng_key, classic_params, stein_params):
            params = {**classic_params, **stein_params}
            loss_val = self.loss.loss(rng_key, params, handlers.scale(self.model, self.loss_temperature), self.guide, *args, **kwargs, **self.static_kwargs)
            return - loss_val

        rng_keys = jax.random.split(rng_key, self.num_stein_particles)
        # loss, particle_ljp_grads = jax.value_and_grad(lambda ps: jfp_fn(rng_keys, self.constrain_fn(model_uparams), self.constrain_fn(unravel_pytree(ps))))(guide_particles)
        loss, particle_ljp_grads = jax.vmap(lambda rk, ps: jax.value_and_grad(lambda ps: 
                                            scaled_loss(rk, self.constrain_fn(classic_uparams), self.constrain_fn(unravel_pytree(ps))))(ps))(rng_keys, stein_particles)
        classic_param_grads = jax.vmap(lambda rk, ps: jax.grad(lambda cps: 
                                            scaled_loss(rk, self.constrain_fn(cps), self.constrain_fn(unravel_pytree(ps))))(classic_uparams))(rng_keys, stein_particles)
        classic_param_grads = tree_map(jax.partial(np.mean, axis=0), classic_param_grads)
        # 3. Calculate kernel on monolithic particle
        kernel = self.kernel_fn.compute(stein_particles)
        # 4. Calculate the attractive force and repulsive force on the monolithic particles
        attractive_force = jax.vmap(lambda y: np.sum(jax.vmap(lambda x, x_ljp_grad: kernel(x, y) * x_ljp_grad)(stein_particles, particle_ljp_grads), axis=0))(stein_particles)
        repulsive_force = jax.vmap(lambda y: np.sum(jax.vmap(lambda x: self.repulsion_temperature * self._kernel_grad(kernel, x, y))(stein_particles), axis=0))(stein_particles)
        particle_grads = (attractive_force + repulsive_force) / self.num_stein_particles
        # 5. Decompose the monolithic particle forces back to concrete parameter values
        stein_param_grads = unravel_pytree_batched(particle_grads)
        # 6. Return loss and gradients (based on parameter forces)
        res_grads = tree_map(lambda x: -x, {**classic_param_grads, **stein_param_grads})
        return -np.mean(loss), res_grads

    
    def init(self, rng_key, *args, **kwargs):
        """
        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: initial :data:`SVGDState`
        """
        rng_key, model_seed, guide_seed = jax.random.split(rng_key, 3)
        model_init = handlers.seed(self.model, model_seed)
        guide_init = handlers.seed(self.guide, guide_seed)
        guide_trace = handlers.trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = handlers.trace(model_init).get_trace(*args, **kwargs, **self.static_kwargs)
        rng_key, particle_seed = jax.random.split(rng_key)
        particle_seeds = jax.random.split(particle_seed, num=self.num_stein_particles)
        self.guide.find_params(particle_seeds, *args, **kwargs, **self.static_kwargs) # Get parameter values for each particle
        params = {}
        inv_transforms = {}
        guide_param_names = set()
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                if site['name'] in self.guide.init_params:
                    pval, _ = self.guide.init_params[site['name']]
                else:
                    pval =  site['value']
                params[site['name']] = transform.inv(pval)
                if site['name'] in guide_trace:
                    guide_param_names.add(site['name'])
        self.guide_param_names = guide_param_names
        self.constrain_fn = jax.partial(transform_fn, inv_transforms)
        return SVGDState(self.optim.init(params), rng_key)

    def get_params(self, state):
        """
        Gets values at `param` sites of the `model` and `guide`.
        :param svi_state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(state.optim_state))
        return params

    def update(self, state, *args, **kwargs):
        """
        Take a single step of SVGD (possibly on a batch / minibatch of data),
        using the optimizer.
        :param state: current state of SVGD.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(state, loss)`.
        """
        rng_key, rng_key_step = jax.random.split(state.rng_key)
        params = self.optim.get_params(state.optim_state)
        loss_val, grads = self._svgd_loss_and_grads(rng_key_step, params, 
                                                    *args, **kwargs, **self.static_kwargs)
        optim_state = self.optim.update(grads, state.optim_state)
        return SVGDState(optim_state, rng_key), loss_val

    def evaluate(self, state, *args, **kwargs):
        """
        Take a single step of SVGD (possibly on a batch / minibatch of data).
        :param state: current state of SVGD.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate loss given the current parameter values (held within `state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given a state
        _, rng_key_eval = jax.random.split(state.rng_key)
        params = self.get_params(state)
        loss_val, _ = self._svgd_loss_and_grads(rng_key_eval, params, 
                                                *args, **kwargs, **self.static_kwargs)
        return loss_val
