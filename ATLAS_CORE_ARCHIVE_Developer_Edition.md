# ATLAS-CORE-ARCHIVE — Developer Edition
**Concept-Atlas — Morphic Graphing Framework**

## 0. Purpose

This document gives developers everything needed to continue the
Morphic Graphing project: the core algorithm, tested Python code,
examples, figures, and contributor notes. Code is MIT-licensed.

## 1. Core Morphic Graphing Algorithm (summary)

1) Define two relations (or one): f(x,y), g(x,y); optionally a coupling
   F(f,g,x,y).  
2) Create a numerical grid for (x,y).  
3) Evaluate the surface: z = z_direct(x,y) or z = F(f,g,x,y).  
4) Compute derivatives (finite differences) for gradients/curvature.  
5) Detect features automatically:
   - Ridge → |∂z/∂x| ≤ ε and ∂²z/∂x² < 0
   - Valley → |∂z/∂x| ≤ ε and ∂²z/∂x² > 0
   - Fault → |∇z| above a high quantile (e.g., 97th)  
6) Slice and analyse sections (constant x or y; oblique cuts).  
7) Interpret the geometry in context (transition, degeneracy, bridge).

This pipeline is identical for any equation. Only z(x,y) (or f,g,F)
changes.

## 2. Python implementation — `morphic_core.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

Func = Callable[[np.ndarray, np.ndarray], np.ndarray]
Coupling = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]

@dataclass
class Grid:
    x: np.ndarray; y: np.ndarray; X: np.ndarray; Y: np.ndarray
    @staticmethod
    def make(x_range: Tuple[float, float], y_range: Tuple[float, float],
             nx: int = 160, ny: int = 160) -> "Grid":
        x = np.linspace(*x_range, nx)
        y = np.linspace(*y_range, ny)
        X, Y = np.meshgrid(x, y)
        return Grid(x, y, X, Y)

@dataclass
class Surface:
    grid: Grid
    Z: np.ndarray
    meta: Dict | None = None

def evaluate_surface(grid: Grid, f: Func | None = None, g: Func | None = None,
                     F: Coupling | None = None, z_direct: Func | None = None) -> Surface:
    X, Y = grid.X, grid.Y
    if z_direct is not None:
        Z = z_direct(X, Y)
    else:
        if f is None or g is None or F is None:
            raise ValueError("Provide either z_direct or (f, g, F).")
        Z = F(f(X, Y), g(X, Y), X, Y)
    return Surface(grid=grid, Z=Z, meta={})

@dataclass
class Derivatives:
    zx: np.ndarray; zy: np.ndarray; zxx: np.ndarray; zyy: np.ndarray; grad_norm: np.ndarray

def derivatives(surf: Surface) -> Derivatives:
    Z = surf.Z
    dx = np.diff(surf.grid.x).mean()
    dy = np.diff(surf.grid.y).mean()
    zx = np.zeros_like(Z); zy = np.zeros_like(Z)
    zx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / (2 * dx)
    zy[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / (2 * dy)
    zx[:, 0] = (Z[:, 1] - Z[:, 0]) / dx
    zx[:, -1] = (Z[:, -1] - Z[:, -2]) / dx
    zy[0, :] = (Z[1, :] - Z[0, :]) / dy
    zy[-1, :] = (Z[-1, :] - Z[-2, :]) / dy
    zxx = np.zeros_like(Z); zyy = np.zeros_like(Z)
    zxx[:, 1:-1] = (Z[:, 2:] - 2 * Z[:, 1:-1] + Z[:, :-2]) / (dx * dx)
    zyy[1:-1, :] = (Z[2:, :] - 2 * Z[1:-1, :] + Z[:-2, :]) / (dy * dy)
    zxx[:, 0] = zxx[:, 1]; zxx[:, -1] = zxx[:, -2]
    zyy[0, :] = zyy[1, :]; zyy[-1, :] = zyy[-2, :]
    grad_norm = np.hypot(zx, zy)
    return Derivatives(zx=zx, zy=zy, zxx=zxx, zyy=zyy, grad_norm=grad_norm)

@dataclass
class Features:
    ridge_mask: np.ndarray; valley_mask: np.ndarray; fault_mask: np.ndarray; stats: Dict

def features(surf: Surface, deriv: Derivatives, ridge_tol: float = 1e-2,
             fault_quantile: float = 0.97) -> Features:
    ridge_mask = (np.abs(deriv.zx) <= ridge_tol) & (deriv.zxx < 0)
    valley_mask = (np.abs(deriv.zx) <= ridge_tol) & (deriv.zxx > 0)
    thr = np.quantile(deriv.grad_norm, fault_quantile)
    fault_mask = deriv.grad_norm >= thr
    stats = {
        "ridge_frac": float(ridge_mask.mean()),
        "valley_frac": float(valley_mask.mean()),
        "fault_thr": float(thr),
        "grad_norm_mean": float(deriv.grad_norm.mean()),
    }
    return Features(ridge_mask, valley_mask, fault_mask, stats)

def slice_at_y(surf: Surface, y_value: float):
    j = int(np.argmin(np.abs(surf.grid.y - y_value)))
    return surf.grid.x, surf.Z[j, :].copy()

def slice_at_x(surf: Surface, x_value: float):
    i = int(np.argmin(np.abs(surf.grid.x - x_value)))
    return surf.grid.y, surf.Z[:, i].copy()

def compute_morphic_surface(x_range, y_range, nx, ny, *,
                            z_direct=None, f=None, g=None, F=None,
                            ridge_tol=1e-2, fault_quantile=0.97):
    grid = Grid.make(x_range, y_range, nx, ny)
    surf = evaluate_surface(grid, f=f, g=g, F=F, z_direct=z_direct)
    deriv = derivatives(surf)
    feats = features(surf, deriv, ridge_tol=ridge_tol, fault_quantile=fault_quantile)
    return {"grid": grid, "surface": surf, "derivatives": deriv, "features": feats}
