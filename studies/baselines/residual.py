"""Training Leaves Traces equations 3–6/B.1, complete MLP branches only."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def signature(down, up):
    down, up = np.asarray(down, dtype=np.float64), np.asarray(up, dtype=np.float64)
    if (down.ndim != 2 or up.ndim != 2 or down.shape[1] != up.shape[0]
            or down.shape[0] != up.shape[1] or down.shape[0] < 2
            or not np.isfinite(down).all() or not np.isfinite(up).all()):
        raise ValueError('Invalid complete MLP factors')
    product = down @ up
    norm, trace = np.linalg.norm(product), np.trace(product)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Zero or nonfinite product')
    concentration = abs(trace) / norm
    product.flat[::len(product) + 1] -= trace / len(product)
    norm = np.linalg.norm(product)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Zero or nonfinite centered signature')
    return product.ravel() / norm, concentration


def score(reference, suspect):
    a, sa = (np.asarray(x, dtype=np.float64) for x in reference)
    b, sb = (np.asarray(x, dtype=np.float64) for x in suspect)
    if (a.ndim != 2 or a.shape != b.shape or len(a) == 0
            or sa.shape != (len(a),) or sb.shape != sa.shape
            or not all(np.isfinite(x).all() for x in (a, b, sa, sb))
            or sa.min() <= 0 or sb.min() < 0):
        raise ValueError('Incompatible or invalid branches/concentration')
    tau = sa.min()
    gate = np.minimum(np.minimum(sa[:, None] / tau, sb[None, :] / tau), 1)
    similarities = np.clip(a @ b.T, -1, 1) * gate
    rows, columns = linear_sum_assignment(similarities, maximize=True)
    return float(similarities[rows, columns].mean()), columns
