import jax.numpy as np

def rbf_kernel(particles):
    dists = np.expand_dims(particles, axis=0) - np.expand_dims(particles, axis=1) # N x N x D
    bandwidth = np.median(np.abs(dists)) ** 2 / np.log(dists.shape[0]) + 1e-5
    def log_kernel(x, y):
        return - (np.linalg.norm(x - y, ord=2) ** 2) / bandwidth
    return log_kernel
