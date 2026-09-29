"""Dense delay-space regression versus SINDy benchmark package."""

from .models import PolynomialVectorField, fit_ddsr, fit_sindy_stlsq

__all__ = ["PolynomialVectorField", "fit_ddsr", "fit_sindy_stlsq"]

