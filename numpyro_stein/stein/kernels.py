from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as np
import jax.scipy.stats

class SteinKernel(ABC):
    @abstractmethod
    def mode(self):
        pass

    @abstractmethod
    def compute(self, particles):
        pass

class RBFKernel(SteinKernel):
    """
    Calculates the Gaussian RBF kernel function with median bandwidth
    :param mode: Either 'norm' (default) specifying to take the norm of each particle or 'vector' to return a component-wise kernel
    """
    def __init__(self, mode='norm', bandwidth_factor: Callable[[float], float]=lambda n: 1 / np.log(n)):
        assert mode == 'norm' or mode == 'vector'
        self._mode = mode
        self.bandwidth_factor = bandwidth_factor

    def compute(self, particles: np.ndarray):
        """
        Computes the kernel function given the input Stein particles
        :param particles: The Stein particles to compute the kernel from
        """
        if self._mode == 'norm' and particles.ndim >= 2:
            particles = np.linalg.norm(particles, ord=2, axis=-1) # N x D -> N
        dists = np.expand_dims(particles, axis=0) - np.expand_dims(particles, axis=1) # N x N (x D)
        dists = np.reshape(dists, (dists.shape[0] * dists.shape[1], -1)) # N * N (x D)
        factor = self.bandwidth_factor(particles.shape[0])
        median = np.argsort(np.linalg.norm(np.abs(dists), ord=2, axis=-1))[int(dists.shape[0] / 2)]
        bandwidth = np.abs(dists)[median] ** 2 * factor + 1e-5
        if self._mode == 'norm':
            bandwidth = bandwidth[0]
        def kernel(x, y):
            if self._mode == 'norm':
                return np.exp (- (np.linalg.norm(x - y, ord=2) ** 2) / bandwidth)
            else:
                return np.exp (- (x - y) ** 2 / bandwidth)
        return kernel

    def mode(self):
        return self._mode
