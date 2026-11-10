"""Small local replacements for midpoint/formatting helpers, not the original utils."""
import numpy as np


def ave(values):
    """Adjacent midpoints along the first axis."""
    values = np.asarray(values)
    return (values[:-1] + values[1:]) / 2


def neat_float(value, digits=6):
    return f"{value:.{digits}g}"


def latex_float(value, digits=3):
    text = neat_float(value, digits)
    if 'e' not in text:
        return text
    mantissa, exponent = text.split('e')
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"
