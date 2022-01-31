"""Configuration suggester for atomistic simulations."""
from random import randrange, random

from ase import Atoms

from .configuration_suggester import ConfigurationSuggester


class AtomDisplacer(ConfigurationSuggester):
    """
    Configuration suggester for atomistic simulations.

    Displaces a random atom in an ASE atoms object by a random vector.
    """

    def __init__(self, maximum_displacement, enforce_pbc=True):
        super(AtomDisplacer, self).__init__()
        self.maximum_displacement = maximum_displacement
        self.enforce_pbc = enforce_pbc

    @staticmethod
    def _enforce_pbc(atoms):
        """
        Explictly enforeces the PBC on an ASE atoms object.

        QE (and potentially other codes?) do that internally. Meaning that the
        raw positions of atoms (in Angstrom) can lie outside of the unit cell.
        When setting up the DFT calculation, these atoms get shifted into
        the unit cell. Since we directly use these raw positions for the
        descriptor calculation, we need to enforce that in the ASE atoms
        objects, the atoms are explicitly in the unit cell.

        Parameters
        ----------
        atoms : ase.atoms
            The ASE atoms object for which the PBC need to be enforced.

        Returns
        -------
        new_atoms : ase.atoms
            The ASE atoms object for which the PBC have been enforced.
        """
        new_atoms = atoms.copy()
        new_atoms.set_scaled_positions(new_atoms.get_scaled_positions())
        return new_atoms

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
        new_configuration = old_configuration.copy()

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
        if self.enforce_pbc:
            new_configuration = AtomDisplacer._enforce_pbc(new_configuration)

        # Ensure there is only one configuration across nodes.
        new_configuration = self.collect_configuration(new_configuration)
        return new_configuration

    def get_info(self):
        """
        Access a dictionary with identifying information.

        Returns
        -------
        info : dict

        """
        info = {}
        info["name"] = type(self).__name__
        info["maximum_displacement"] = self.maximum_displacement
        info["enforce_pbc"] = self.enforce_pbc
        return info

