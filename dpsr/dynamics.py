"""Shared explicit integration and within-system forecast metrics."""
import numpy as np


def model_rhs(model, state):
    return np.asarray(model.predict(np.asarray(state)[None, :]))[0]


def rk4_step(model, state, dt):
    k1 = model_rhs(model, state)
    k2 = model_rhs(model, state + dt*k1/2)
    k3 = model_rhs(model, state + dt*k2/2)
    k4 = model_rhs(model, state + dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def cumulative_error_horizon(truth, prediction, dt, tolerance, *, coordinate=None):
    if prediction is None:
        return np.nan
    count = min(len(truth), len(prediction))
    error = np.asarray(truth[:count]) - np.asarray(prediction[:count])
    squared = error[:, coordinate]**2 if coordinate is not None else np.mean(error**2, axis=1)
    crossings = np.flatnonzero(np.cumsum(squared)*dt > tolerance)
    return float(crossings[0]*dt if len(crossings) else (count-1)*dt)


def nrmse(truth, prediction, *, coordinate=None):
    if prediction is None:
        return np.nan
    count = min(len(truth), len(prediction))
    truth, prediction = np.asarray(truth[:count]), np.asarray(prediction[:count])
    if coordinate is not None:
        truth, prediction = truth[:, coordinate], prediction[:, coordinate]
    return float(np.sqrt(np.mean((truth-prediction)**2))/(np.std(truth)+1e-12))
