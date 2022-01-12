"""Configuration suggester for atomistic simulations."""
from copy import deepcopy
from random import randrange, random

from ase import Atoms

from .configuration_suggester import ConfigurationSuggester


class AtomDisplacer(ConfigurationSuggester):
    """
    Configuration suggester for atomistic simulations.

    Displaces a random atom in an ASE atoms object by a random vector.
    """

    def __init__(self, maximum_displacement):
        super(AtomDisplacer, self).__init__()
        self.maximum_displacement = maximum_displacement

    def suggest_new_configuration(self, old_configuration: Atoms):
        """
        Suggest a new configuration based on the old.

        Parameters
        ----------
        old_configuration: ase.Atoms
            Old configuration based on which a new one will be suggested.

        Returns
        -------
        new_configuration: ase.Atoms
            New configuration as suggested by algorithm.

        """
        new_configuration = deepcopy(old_configuration)

        # Which atom do we want to displace?
        number_atom = randrange(0, len(old_configuration))

        # In which direction to we want to displace?
        # 0: positive x, 1: negative x
        # 2: positve y, 3: negative y
        # 4: positive z, 5: positive z
        direction = randrange(0, 6)

        # How far do we want to displace the atom?
        displacement = random()*self.maximum_displacement
        if direction % 2 == 1:
            displacement *= -1
            direction -= 1

        # Displace the atom.
        positions = new_configuration.get_positions()
        positions[number_atom, direction//2] += displacement
        new_configuration.set_positions(positions)
        return new_configuration

