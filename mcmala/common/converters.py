import scipy.constants
import ase.units


def kelvin_to_rydberg(temperature_K):
    """
    Convert a temperature from Kelvin to Rydberg energy units.

    Parameters
    ----------
    temperature_K : float
        Temperature in Kelvin.

    Returns
    -------
    temperature_Ry : float
        Temperature expressed in Rydberg.

    """
    k_B = scipy.constants.Boltzmann
    Ry_in_Joule = scipy.constants.Rydberg*scipy.constants.h*scipy.constants.c
    return (k_B*temperature_K)/Ry_in_Joule
