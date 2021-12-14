"""
Evaluator and suggester for Ising Model.
"""
from random import randrange
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from .evaluator import Evaluator
from .configuration_suggester import ConfigurationSuggester


class IsingGrid:
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


class IsingModelEvaluator(Evaluator):
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

    def get_total_energy(self, configuration: IsingGrid):
        energy = 0.0
        for i in range(0, configuration.lattice_size):
            for j in range(0, configuration.lattice_size):
                energy += self.__get_local_hamiltonian(configuration, i, j)
        return -0.5*energy*self.interaction_strength


class IsingModelConfigurations(ConfigurationSuggester):
    def __init__(self):
        super(IsingModelConfigurations, self).__init__()

    def suggest_new_configuration(self, old_configuration: IsingGrid):
        new_configuration = deepcopy(old_configuration)
        pos_to_flip = randrange(new_configuration.lattice_size *
                                new_configuration.lattice_size)
        x_to_flip = pos_to_flip // old_configuration.lattice_size
        y_to_flip = pos_to_flip % old_configuration.lattice_size
        new_configuration.lattice[x_to_flip, y_to_flip] *= -1
        return new_configuration

