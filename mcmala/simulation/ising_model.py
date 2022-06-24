"""Evaluator, suggester, configurations for the Ising model."""
from random import randrange
from copy import deepcopy

import numpy as np
matplotlib_avail = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_avail = False
    pass

from ase.calculators.calculator import Calculator
from .configuration_suggester import ConfigurationSuggester


class IsingGrid:
    """
    Represents a grid of spins.

    Such a grid is at the center of the Ising model. Spins can be either
    +1 or -1 (spin up/spin down). Only a quadratic 2D lattice is supported.

    Parameters
    ----------
    lattice_size : int
        Size of the lattice (in either direction, a quadratic lattice is
                         assumed.)
    initType : string
        Type of initialization to be performed. Default "random", assgning
        spins at random. "positive" or "negative" initializes the lattice
        entirely with positive or negative spins, respectively.
    """

    def __init__(self, lattice_size, initType="random"):
        self.lattice_size = lattice_size
        self.lattice = np.zeros((self.lattice_size, self.lattice_size),
                                dtype=np.int8)
        for i in range(0, self.lattice_size):
            for j in range(0, self.lattice_size):
                if initType == "random":
                    randomNumber = randrange(2)
                    if randomNumber == 0:
                        randomNumber = -1
                    self.lattice[i, j] = randomNumber
                elif initType == "negative":
                    self.lattice[i, j] = -1
                elif initType == "positive":
                    self.lattice[i, j] = 1
                else:
                    raise Exception("Unknown init type")

    def visualize(self, ax=None):
        """
        Visualize an Ising model spin grid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axis to be used for plotting. If "None", one will be created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axis to be used for plotting. If one was provided,
            the plot will have been added to this one.

        """
        if matplotlib_avail:
            if ax is None:
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(1, 1, 1)
            plusx = []
            plusy = []
            minusx = []
            minusy = []
            for i in range(0, self.lattice_size):
                for j in range(0, self.lattice_size):
                    if self.lattice[i,j] == 1:
                        plusx.append(i)
                        plusy.append(j)
                    else:
                        minusx.append(i)
                        minusy.append(j)

            ax.scatter(plusx, plusy, label="plus", marker="s",
                       color="tab:red", s=470)
            ax.scatter(minusx, minusy, label="minus", marker="s",
                       color="tab:blue", s=470)

            return ax
        else:
            raise Exception("No matplotlib found, cannot visualize Ising "
                            "grid.")


class IsingModelEvaluator(Calculator):
    """
    Evaluator for the Ising model.

    An object of this class can calculate the total energy of an Ising
    model spin grid.

    Parameters
    ----------
    interaction_strength : float
        Interaction strength of Ising model, in eV.

    """

    def __init__(self, interaction_strength):
        super(IsingModelEvaluator, self).__init__()
        self.interaction_strength = interaction_strength

    @staticmethod
    def __get_local_hamiltonian(configuration: IsingGrid, pointX,
                                pointY):
        thisPoint = configuration.lattice[pointX, pointY]
        localHamiltonian = 0.0
        # We assume periodic boundary conditions.
        if pointX == 0:
            localHamiltonian += configuration.lattice[
                                    configuration.lattice_size-1, pointY] \
                                * thisPoint
        else:
            localHamiltonian += configuration.lattice[pointX - 1, pointY] \
                                * thisPoint

        if pointY == 0:
            localHamiltonian += configuration.lattice[
                                    pointX, configuration.lattice_size-1] \
                                * thisPoint
        else:
            localHamiltonian += configuration.lattice[pointX, pointY - 1] \
                                * thisPoint

        if pointX == (configuration.lattice_size-1):
            localHamiltonian += configuration.lattice[0, pointY] * thisPoint
        else:
            localHamiltonian += configuration.lattice[pointX + 1, pointY] \
                                * thisPoint

        if pointY == (configuration.lattice_size-1):
            localHamiltonian += configuration.lattice[pointX, 0] * thisPoint
        else:
            localHamiltonian += configuration.lattice[pointX, pointY + 1] \
                                * thisPoint
        return localHamiltonian

    def calculate(self, atoms: IsingGrid):
        """
        Calculate the total energy of a spin grid.

        Parameters
        ----------
        atoms : IsingGrid
            Spin grid based on which the total energy will be calculated.

        Returns
        -------
        total_energy : float
            Total energy of the given configuration, in eV.
        """
        self.results["energy"] = 0.0
        energy = 0.0
        for i in range(0, atoms.lattice_size):
            for j in range(0, atoms.lattice_size):
                energy += self.__get_local_hamiltonian(atoms, i, j)
        self.results["energy"] = -0.5*energy*self.interaction_strength

    def calculate_properties(self, atoms: IsingGrid, properties):
        """
        After a calculation, calculate additional properties.

        This is separate from the calculate function because of
        MALA-MC simulations. For these energy and additional property
        calculation need to be separate.
        """
        pass


class IsingModelConfigurations(ConfigurationSuggester):
    """
    Configuration suggester for the Ising model.

    Randomly flips a spin.
    """

    def __init__(self):
        super(IsingModelConfigurations, self).__init__()

    def suggest_new_configuration(self, old_configuration: IsingGrid):
        """
        Suggest a new configuration for an Ising model spin grid.

        This is done by randomly flipping a spin.

        Parameters
        ----------
        old_configuration: IsingGrid
            Spin grid for which to suggest a new spin grid.

        Returns
        -------
        new_configuration: IsingGrid
            New spin grid.

        """
        new_configuration = deepcopy(old_configuration)
        pos_to_flip = randrange(new_configuration.lattice_size *
                                new_configuration.lattice_size)
        x_to_flip = pos_to_flip // old_configuration.lattice_size
        y_to_flip = pos_to_flip % old_configuration.lattice_size
        new_configuration.lattice[x_to_flip, y_to_flip] *= -1
        return new_configuration

    def to_json(self):
        """
        Convert object into JSON seriazable content.

        Returns
        -------
        info : dict
            Info that can be saved to a dict.

        """
        info = {"name": type(self).__name__}
        return info
