import numpy as np


def generate_equation_system(_a, _b, _c, _d, _I):
    def _equation(w, t):
        dvdt = 0.04 * w[0] * w[0] + 5 * w[0] + 140 - w[1] + _I
        dudt = _a * (_b * w[0] - w[1])

        return np.array([dvdt, dudt])

    return _equation


def generate_reset_condition(c, d, threshold=30):
    def _condition(w):
        if w[0] >= threshold:
            return np.array([c, d + w[1]])
        else:
            return w

    return _condition
