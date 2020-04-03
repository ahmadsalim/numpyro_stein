from functools import namedtuple
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import transform_fn
from autoguides import AutoDelta

import jax
import jax.random

SVGDState = namedtuple('SVGDState', ['optim_state', 'guide_param_name', 'rng_key'])

class SVGD(object):
    """
    Stein Variational Gradient Descent for Non-parametric Inference.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param num_particles: number of particles for Stein inference.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """
    def __init__(self, model, guide, optim, num_particles=10, **static_kwargs):
        assert isinstance(guide, AutoDelta) # Only AutoDelta guide supported for now
        self.model = model
        self.guide = guide
        self.optim = optim
        self.static_kwargs = static_kwargs
        self.num_particles = num_particles
        self.constrain_fn = None
    
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
        rng_key, particle_seeds = jax.random.split(rng_key, 1 + self.num_particles)
        # TODO Fix this so that we can run guide with different parameter values
        guide_particle_params = self.guide.find_params(particle_seeds, *args, **kwargs, **self.static_kwargs) # Get parameter values for each particle
        params = {}
        inv_transforms = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                pval = guide_particle_params.get(site['name'], site['value'])
                params[site['name']] = transform.inv(pval)

        self.constrain_fn = jax.partial(transform_fn, inv_transforms)
        return SVGDState(self.optim.init(params), guide_particle_params.keys(), rng_key)
