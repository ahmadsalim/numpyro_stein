from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as onp
import numpy.random as onpr
import jax.numpy as np
import jax.scipy.stats

class SteinKernel(ABC):
    @abstractmethod
    def mode(self):
        """
        Returns the type of kernel, either 'norm' or 'vector' or 'matrix'.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, particles):
        """
        Computes the kernel function given the input Stein particles
        :param particles: The Stein particles to compute the kernel from
        """
        raise NotImplementedError

class RBFKernel(SteinKernel):
    """
    Calculates the Gaussian RBF kernel function with median bandwidth.
    This is the kernel used in the original "Stein Variational Gradient Descent" paper by Liu and Wang
    :param mode: Either 'norm' (default) specifying to take the norm of each particle or 'vector' to return a component-wise kernel
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n (default 1/log(n))
    """
    def __init__(self, mode='norm', bandwidth_factor: Callable[[float], float]=lambda n: 1 / np.log(n)):
        assert mode == 'norm' or mode == 'vector'
        self._mode = mode
        self.bandwidth_factor = bandwidth_factor

    def compute(self, particles: np.ndarray):
        if self._mode == 'norm' and particles.ndim >= 2:
            particles = np.linalg.norm(particles, ord=2, axis=-1) # N x D -> N
        dists = np.expand_dims(particles, axis=0) - np.expand_dims(particles, axis=1) # N x N (x D)
        dists = np.reshape(dists, (dists.shape[0] * dists.shape[1], -1)) # N * N x D
        factor = self.bandwidth_factor(particles.shape[0])
        median = np.argsort(np.linalg.norm(np.abs(dists), ord=2, axis=-1))[int(dists.shape[0] / 2)]
        bandwidth = np.abs(dists)[median] ** 2 * factor + 1e-5
        if self._mode == 'norm':
            bandwidth = bandwidth[0]
        def kernel(x, y):
            diff = np.linalg.norm(x - y, ord=2) if self._mode == 'norm' and x.ndim >= 1 else x - y
            return np.exp (- diff ** 2 / bandwidth)
        return kernel

    def mode(self):
        return self._mode

# TODO Test kernel
class IMQKernel(SteinKernel):
    """
    Calculates the IMQ kernel, from "Measuring Sample Quality with Kernels" by Gorham and Mackey
    :param mode: Either 'norm' (default) specifying to take the norm of each particle or 'vector' to return a component-wise kernel
    :param const: Positive multi-quadratic constant (c)
    :param exponent: Inverse exponent (beta) between (-1, 0)
    """
    # Based on 
    def __init__(self, mode='norm', const=1.0, expon=-0.5):
        assert mode == 'norm' or mode == 'vector'
        assert 0.0 < const 
        assert -1.0 < expon < 0.0
        self._mode = mode
        self.const = const
        self.expon = expon

    def mode(self):
        return self._mode

    def compute(self, particles: np.ndarray):
        def kernel(x, y):
            diff = np.linalg.norm(x - y, ord=2) if self._mode == 'norm' else x - y
            return (np.array(self.const) ** 2 + diff ** 2) ** self.expon
        return kernel

# TODO Test kernel
class LinearKernel(SteinKernel):
    """
    Calculates the linear kernel, from "Stein Variational Gradient Descent as Moment Matching" by Liu and Wang
    """
    def __init__(self):
        self._mode = 'norm'

    def mode(self):
        return self._mode

    def compute(self, particles: np.ndarray):
        def kernel(x, y):
            if x.ndim >= 1:
                return x @ y + 1
            else:
                return x * y + 1
        return kernel

# TODO Test kernel
class RandomFeatureKernel(SteinKernel):
    """
    Calculates the random kernel, from "Stein Variational Gradient Descent as Moment Matching" by Liu and Wang
    :param bandwidth_subset: How many particles should be used to calculate the bandwidth? (default None, meaning all particles)
    :param random_indices: The set of indices which to do random feature expansion on. (default None, meaning all indices)
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n (default 1/log(n))
    """
    def __init__(self, bandwidth_subset=None, random_indices=None, bandwidth_factor: Callable[[float], float]=lambda n: 1 / np.log(n)):
        assert bandwidth_subset is None or bandwidth_subset > 0
        self._mode = 'norm'
        self.bandwidth_subset = bandwidth_subset
        self.random_indices = None
        self.bandwidth_factor = bandwidth_factor
        self._random_weights = None
        self._random_biases = None

    def mode(self):
        return self._mode

    def compute(self, particles: np.ndarray):
        if self._random_weights is None:
            self._random_weights = np.array(onpr.randn(*particles.shape))
            self._random_biases = np.array(onpr.rand(*particles.shape) * 2 * onp.pi)
        factor = self.bandwidth_factor(particles.shape[0])
        if particles.ndim >= 2:
            particles = np.linalg.norm(particles, ord=2, axis=-1) # N x D -> N
        if self.bandwidth_subset is not None:
            particles = particles[onpr.choice(particles.shape[0], self.bandwidth_subset)]
        dists = np.expand_dims(particles, axis=0) - np.expand_dims(particles, axis=1) # N x N
        dists = np.reshape(dists, (dists.shape[0] * dists.shape[1], -1)) # N * N x 1
        median = np.argsort(np.linalg.norm(np.abs(dists), ord=2, axis=-1))[int(dists.shape[0] / 2)]
        bandwidth = np.abs(dists)[median] ** 2 * factor + 1e-5
        def feature(x, w, b):
            return np.sqrt(2) * np.cos((x @ w + b) / bandwidth)
        def kernel(x, y):
            ws = self._random_weights if self.random_indices is None else self._random_weights[self.random_indices]
            bs = self._random_biases if self.random_indices is None else self._random_biases[self.random_indices]
            return np.sum(jax.vmap(lambda w, b: feature(x, w, b) * feature(y, w, b))(ws, bs))
        return kernel

class MixtureKernel(SteinKernel):
    """
    Implements a mixture of multiple kernels
    :param ws: Weight of each kernel in the mixture
    :param kernel_fns: Different kernel functions to mix together
    """
    def __init__(self, ws: List[float], kernel_fns: List[SteinKernel]):
        assert len(ws) == len(kernel_fns)
        assert len(kernel_fns) > 1
        assert all(kf.mode() == kernel_fns[0].mode() for kf in kernel_fns)
        self.ws = ws
        self.kernel_fns = kernel_fns
    
    def mode(self):
        return self.kernel_fns[0].mode()

    def compute(self, particles: np.ndarray):
        kernels = [kf.compute(particles) for kf in self.kernel_fns]
        def kernel(x, y):
            res = self.ws[0] * kernels[0](x, y)
            for w, k in zip(self.ws[1:], kernels[1:]):
                res = res + w * k(x, y)
            return res
        return kernel
