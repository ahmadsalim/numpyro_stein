from contextlib import ExitStack  # python 3
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro import optim
from numpyro.distributions.transforms import biject_to
from numpyro.infer import SVI, ELBO
from numpyro.infer.util import find_valid_initial_params, init_to_uniform
from numpyro.contrib.autoguide import AutoGuide

class PlatedAutoGuide(AutoGuide):
    def __init__(self, model, *, prefix='auto', create_plates=None):
        self.create_plates = create_plates
        self._prototype_frames = {}
        super(PlatedAutoGuide, self).__init__(model, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(PlatedAutoGuide, self)._setup_prototype(*args, **kwargs)
        for name, site in self.prototype_trace.items():
            if site['type'] != 'sample' or site['is_observed']:
                continue
            for frame in site['cond_indep_stack']:
                if frame.vectorized:
                    self._prototype_frames[frame.name] = frame
                else:
                    raise NotImplementedError("AutoGuide does not support sequential numpyro.plate")

    def _create_plates(self, *args, **kwargs):
        if self.create_plates is None:
            self.plates = {}
        else:
            plates = self.create_plates(*args, **kwargs)
            if isinstance(plates, numpyro.plate):
                plates = [plates]
            assert all(isinstance(p, numpyro.plate) for p in plates), \
                "create_plates() returned a non-plate"
            self.plates = {p.name: p for p in plates}
            for name, frame in sorted(self._prototype_frames.items()):
                if name not in self.plates:
                    self.plates[name] = numpyro.plate(name, frame.size, dim=frame.dim)
        return self.plates

class AutoDelta(PlatedAutoGuide):
    def __init__(self, model, *, prefix='auto', init_strategy=init_to_uniform(), create_plates=None):
        self.init_strategy = init_strategy
        self.params = {}
        super(AutoDelta, self).__init__(model, prefix=prefix, create_plates=create_plates)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site['type'] != 'sample' or site['is_observed']:
                continue
            with ExitStack() as stack:
                for frame in site['cond_indep_stack']:
                    stack.enter_context(plates[frame.name])
                if site['intermediates']:
                    event_ndim = len(site['fn'].base_dist.event_shape)
                else:
                    event_ndim = len(site['fn'].event_shape)
                param_name, param_val, constraint = self.params[name]
                val_param = numpyro.param(param_name, param_val, constraint=constraint)
                result[name] = numpyro.sample(name, dist.Delta(val_param, event_ndim=event_ndim))
        return result

    def _sample_latent(self, *args, **kwargs):
        raise NotImplementedError

    def sample_posterior(self, rng_key, *args, **kwargs):
        raise NotImplementedError

    def _setup_prototype(self, *args, **kwargs):
        super(AutoDelta, self)._setup_prototype(*args, **kwargs)
        rng_key = numpyro.sample("_{}_rng_key_init".format(self.prefix), dist.PRNGIdentity())
        init_params, _ = handlers.block(find_valid_initial_params)(rng_key, self.model,
                                                                   init_strategy=self.init_strategy,
                                                                   model_args=args,
                                                                   model_kwargs=kwargs)
        for name, site in self.prototype_trace.items():
            if site['type'] == 'sample' and not site['is_observed']:
                param_name = "{}_{}".format(self.prefix, name)
                param_val = biject_to(site['fn'].support)(init_params[name])
                self.params[name] = (param_name, param_val, site['fn'].support)
                numpyro.param(param_name, param_val, constraint=site['fn'].support)
