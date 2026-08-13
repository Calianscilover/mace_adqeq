from .qeq import JAXQEqModel, QEqParameterPredictor, QEqParameters, QEqResult

__all__ = [
    "JAXQEqModel",
    "MACEJAXQEqCalculator",
    "QEqParameterPredictor",
    "QEqParameters",
    "QEqResult",
]


def __getattr__(name):
    if name == "MACEJAXQEqCalculator":
        from .calculator import MACEJAXQEqCalculator

        return MACEJAXQEqCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
