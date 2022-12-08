"""
Collection of useful observable calculation functions.
"""

from ase.units import m, J
import numpy as np


def stress_to_pressure(stress, convert_to=None):
    if len(np.shape(stress)) == 2:
        pressure = (stress[0, 0] + stress[1, 1] + stress[2, 2]) / -3.0
    elif len(np.shape(stress)) == 3:
        pressure = (stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2]) / \
                   -3.0
    else:
        raise Exception("Invalid stress tensor format.")

    return pressure_unit_conversion(pressure, convert_to)


def pressure_unit_conversion(pressure, convert_to):
    if convert_to == "kbar":
        pressure *= ((m * m * m) / J) / 1e8
    elif convert_to is None:
        pass
    else:
        raise Exception("Invalid pressure units selected.")
    return pressure
