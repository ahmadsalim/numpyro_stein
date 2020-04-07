from collections import namedtuple
import jax
import jax.numpy as np
from jax import lax
from jax.dtypes import canonicalize_dtype
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

pytree_metadata = namedtuple('pytree_metadata', ['flat', 'shape', 'event_size', 'dtype'])


def _ravel_list(*leaves, batch_dims):
    leaves_metadata = tree_map(lambda l: pytree_metadata(
        np.reshape(l, (*np.shape(l)[:batch_dims], -1)), np.shape(l), 
        np.prod(np.shape(l)[batch_dims:], dtype='int32'), canonicalize_dtype(lax.dtype(l))), leaves)
    leaves_idx = np.cumsum(np.array((0,) + tuple(d.event_size for d in leaves_metadata)))

    def unravel_list(arr):
        return [np.reshape(lax.dynamic_slice_in_dim(arr, leaves_idx[i], m.event_size, axis=batch_dims),
                           m.shape).astype(m.dtype)
                for i, m in enumerate(leaves_metadata)]

    flat = np.concatenate([m.flat for m in leaves_metadata], axis=-1) if leaves_metadata else np.array([])
    return flat, unravel_list


def ravel_pytree(pytree, *, batch_dims=0):
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(*leaves, batch_dims=batch_dims)

    def unravel_pytree(arr):
        return tree_unflatten(treedef, unravel_list(arr))

    return flat, unravel_pytree