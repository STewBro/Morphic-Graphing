import numpy as np
from morphic_core import compute_morphic_surface, slice_at_y

# Example 1: Cubic family
def z_cubic(X, A):
    return X**3 - A*X

res_cubic = compute_morphic_surface(
    x_range=(-2, 2), y_range=(-2, 2), nx=200, ny=200,
    z_direct=z_cubic, ridge_tol=5e-3, fault_quantile=0.98
)

# Example 2: Quantum–Relativity toy surface
def R_qm_rel(X, Y):
    eps = 1e-9
    return np.exp(-(np.abs(X) / (1 + 1.0/np.maximum(Y, eps)))**2) * (1 + 1.0/(np.maximum(Y, eps)**2))

res_qm = compute_morphic_surface(
    x_range=(-3, 3), y_range=(1, 10), nx=200, ny=200,
    z_direct=R_qm_rel, ridge_tol=1e-2, fault_quantile=0.97
)

# Guided slices for the QM–Relativity scene
x1, z1 = slice_at_y(res_qm["surface"], 1.0)
x5, z5 = slice_at_y(res_qm["surface"], 5.0)
