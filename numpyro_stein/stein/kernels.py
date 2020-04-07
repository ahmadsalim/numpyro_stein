import jax.numpy as np

def rbf_kernel(particles):
    dists = np.expand_dims(particles, axis=0) - np.expand_dims(particles, axis=1) # N x N x D
    bandwidth = np.median(np.abs(dists)) ** 2 / np.log(dists.shape[0])
    def log_kernel(x, y):
        return - ((x - y) ** 2) - np.log(bandwidth)
    return log_kernel