"""Data-fitted complete quadratic vector fields for the odd KS subspace.

Training uses independent spatial snapshots and exact spectral derivative labels,
not a periodic orbit. Coordinates are invertibly related to interior nodal
values by a discrete sine transform. Reflection equivariance is imposed on the
fitted polynomial; no KS coefficient is inserted into the model.
"""
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dpsr.polynomial import solve_dense_regression
from ks_neumann import NeumannKSConfig, velocity_rhs_sine

@dataclass
class QuadraticSurrogate:
    constant: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray  # output, input, input; symmetric in inputs

    def predict(self, states):
        x = np.asarray(states)
        return self.constant + x @ self.linear.T + (x[..., :, None]*x[..., None, :]).reshape(*x.shape[:-1], -1) @ self.quadratic.reshape(len(self.constant), -1).T

    def jacobian(self, states):
        return self.linear + 2*np.einsum('oij,...j->...oi', self.quadratic, states, optimize=True)

    def save(self, path):
        np.savez(path, constant=self.constant, linear=self.linear, quadratic=self.quadratic)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(*(z[k] for k in ('constant','linear','quadratic')))


def fit_surrogate(modes, seed=173, interval_length=22.):
    """Full-rank least squares from paired random snapshots (no orbit input).

    Pairing x and -x separates even and odd polynomial features. This improves
    conditioning without assuming any values of the polynomial coefficients.
    """
    rng = np.random.default_rng(seed+modes)
    config = NeumannKSConfig(spatial_modes=modes, full_grid_nodes=4*modes,
                             interval_length=interval_length)
    i,j = np.triu_indices(modes)
    count = 2*(len(i)+1)
    scale = 1/(1+np.arange(modes)/4)**2
    z = rng.standard_normal((count,modes))
    x = z*scale
    positive = velocity_rhs_sine(x,config)
    negative = velocity_rhs_sine(-x,config)
    linear_fit, linear_diagnostics = solve_dense_regression(z, (positive-negative)/2, driver='gelsy')
    linear = linear_fit/scale
    features = np.column_stack([np.ones(count),z[:,i]*z[:,j]])
    even_fit, even_diagnostics = solve_dense_regression(features, (positive+negative)/2, driver='gelsy')
    even = even_fit.T
    rank = even_diagnostics['rank']
    if rank != len(i)+1:
        raise RuntimeError('Training library is rank deficient')
    constant = even[0].copy()
    quadratic = np.zeros((modes,modes,modes))
    for p,(a,b) in enumerate(zip(i,j)):
        value = even[p+1]/(scale[a]*scale[b])
        quadratic[:,a,b] = value/(1 if a==b else 2)
        quadratic[:,b,a] = quadratic[:,a,b]
    # R acts on sine mode k by (-1)^k. Average model with R f(R x).
    parity = (-1.)**np.arange(1,modes+1)
    constant *= (1+parity)/2
    linear *= (1+parity[:,None]*parity[None,:])/2
    quadratic *= (1+parity[:,None,None]*parity[None,:,None]*parity[None,None,:])/2
    model = QuadraticSurrogate(constant,linear,quadratic)
    test = rng.standard_normal((512,modes))*scale
    truth = velocity_rhs_sine(test,config)
    error = model.predict(test)-truth
    metrics = dict(spatial_modes=modes, training_snapshots=2*count,
        quadratic_library_rank=int(rank), heldout_snapshots=len(test),
        heldout_rhs_relative_l2=float(np.linalg.norm(error)/np.linalg.norm(truth)),
        heldout_rhs_max_abs=float(np.max(np.abs(error))),
        derivative_labels='exact spectral RHS on independent random spatial snapshots',
        training_periodic_orbits=0)
    return model,metrics
