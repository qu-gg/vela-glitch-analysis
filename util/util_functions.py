"""
@file util_functions.py
@author Ryan Missel

Holds utility functions related to both the VAE and SOM
"""
import uuid
import numpy as np


def get_window(signals):
    """
    Gets a centered window around the average indice of max value for the given dataset,
    using -100 and +100 timesteps around it
    :arg signals: dataset of signals
    """
    center = np.argmax(np.mean(signals, axis=0))

    # Appropriate shifting if peak is near the beginning
    if center - 100 < 0:
        forwards = center + 100 + (100 - center)
        print("=> Window: ({}, {})".format(0, forwards))
        return signals[:, 0:forwards]

    # Shifting if peak occurs at the end
    if center + 100 > 1220:
        print("=> Window: ({}, {})".format(1020, 1220))
        return signals[:, 1020:]

    print("=> Window: ({}, {})".format(center - 100, center + 100))
    return signals[:, center - 100:center + 100]


def get_maxes(recons, cluster_index, som_shape):
    """
    Handles getting the max values across the given clusters
    :arg recons: reconstructed signal dataset
    :arg cluster_index: singal cluster indices
    :arg som_shape: shape of the som
    :return: list of maxes over the clusters
    """
    maxes = []

    # Get cluster-specific metrics
    for idx in range(som_shape[0] * som_shape[1]):
        # Extract subset for both reconstructions and the raw signals
        subset = recons[cluster_index == idx]
        if len(subset) == 0:
            maxes.append(idx + 1)
            continue

        # Get metrics for this subset
        maxs = np.max(subset, axis=1)

        # Append means of subset metrics to arrays
        maxes.append(np.mean(maxs))

    return maxes


def vector_map(data, _map):
    """Remaps numeric values in a data vector.
    :param data: A numpy array of integer or float dtype.
    :param _map: A list of tuple representing the mappings.
                 Alternatively, a dict can be provided directly.
                 ``[(2, 1), (1, 0), (3, -10)]`` would transform all the
                 ``2 -> 1``, ``1 -> 0`` and ``3 -> -10``.
    :returns: A remapped numpy array.
    The strategy used to avoid collisions when sequentially remapping is to
    use transitive mapping. This means that the mapping is done in two steps:
    A mapping to a unique (random) value and then a subsequent mapping to the
    target value.
    This strategy is only used if there are collisions.
    """

    if type(_map) is not dict:
        _map = dict(_map)

    keys = set(_map.keys())
    targets = set(_map.values())

    # Infer the target dtype.
    target_dtype = set([float if np.isnan(t) else type(t) for t in targets])

    if len(target_dtype) != 1:
        raise TypeError("Ambiguous target dtype. Make sure that the provided "
                        "mapper uses consistent type for the second element "
                        "of the tuples.")
    target_dtype = target_dtype.pop()
    out = data.astype(target_dtype)

    for key, target in _map.items():
        if np.isnan(key):
            raise TypeError("Can't use NaNs as mapping keys.")

        if target in keys:
            # There will be a collision, so we need to use the transitive
            # mapping.
            transitive_key = hash(str(uuid.uuid4()))
            out[data == key] = transitive_key
            out[out == transitive_key] = target
        else:
            out[data == key] = target

    return out