from numpyro.distributions import Delta
from numpyro.distributions.util import sum_rightmost
import jax.numpy as np

class Flat(Delta):
    def log_prob(self, value):
        lp = np.zeros_like(value)
        lp = sum_rightmost(value, len(self.event_shape))
        return lp
