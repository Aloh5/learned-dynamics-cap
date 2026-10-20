"""Dense polynomial surrogate reconstruction for learned dynamics and CAP."""
from .polynomial import PolynomialLibrary, PolynomialModel, fit_polynomial_vector_field

__all__ = ['PolynomialLibrary', 'PolynomialModel', 'fit_polynomial_vector_field']
