"""Witness Overlap cosine equations 1/5; stored-weight, equation-level adapter."""
import numpy as np


def moments(a, b, w):
    a, b, w = (np.asarray(x, dtype=np.float64) for x in (a, b, w))
    if (a.ndim != 2 or a.shape != b.shape or a.shape != w.shape or a.size == 0
            or not all(np.isfinite(x).all() for x in (a, b, w))):
        raise ValueError('Invalid aligned weight matrices')
    u, v, t = b-a, w-a, w-b
    return np.array([np.sum(u*u), np.sum(v*v), np.sum(t*t),
                     np.sum(u*v), -np.sum(u*t)])


def anchor_scores(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or (values[:3] <= 0).any():
        raise ValueError('Zero or invalid outgoing update')
    aa, aw, bw, dot_a, dot_b = values
    return (float(np.clip(dot_a / (np.sqrt(aa)*np.sqrt(aw)), -1, 1)),
            float(np.clip(dot_b / (np.sqrt(aa)*np.sqrt(bw)), -1, 1)))


def aggregate(roles):
    if not roles or any(not values for values in roles.values()):
        raise ValueError('Role without eligible layers')
    means = np.array([np.mean(roles[k], axis=0) for k in sorted(roles)])
    if means.ndim != 2 or means.shape[1] != 2 or not np.isfinite(means).all():
        raise ValueError('Invalid per-tensor scores')
    return tuple(float(x) for x in means.mean(axis=0))
