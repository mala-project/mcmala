is_mala_available = True
try:
    import mala
except ModuleNotFoundError:
    is_mala_available = False
from ase.calculators.espresso import Espresso
from ase.calculators.calculator import FileIOCalculator

class EspressoMC(Espresso):
    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='espresso', atoms=None, mala_params=None, **kwargs):
        """
                All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict, all other ``kwargs`` are passed as parameters.

        Accepts all the options for pw.x as given in the QE docs, plus some
        additional options:

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts: (int, int, int), dict, or BandPath
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.
            If ``kpts`` is set to ``None``, only the Γ-point will be included
            and QE will use routines optimized for Γ-point-only calculations.
            Compared to Γ-point-only calculations without this optimization
            (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
            are typically reduced by half.
            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).

        Parameters
        ----------
        restart
        ignore_bad_restart_file
        label
        atoms
        mala_params
        kwargs
        """
        super(EspressoMC, self).__init__(restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.observables_calculator = None
        self.mala_params = mala_params
        if is_mala_available and mala_params is not None:
            self.observables_calculator = mala.LDOS(mala_params)

    def calculate_properties(self, atoms, properties):
        if not is_mala_available or self.observables_calculator is None:
            raise Exception("Cannot calculate additional MC observables "
                            "without MALA present.")

        if "rdf" in properties:
            self.results["rdf"] = self.observables_calculator.\
                get_radial_distribution_function(atoms)
        if "tpcf" in properties:
            self.results["tpcf"] = self.observables_calculator.\
                get_three_particle_correlation_function(atoms)
        if "static_structure_factor" in properties:
            self.results["static_structure_factor"] = self.observables_calculator.\
                get_static_structure_factor(atoms)
        if "ion_ion_energy" in properties:
            # TODO: Implement this, this can easily be read from the output
            # file.
            raise Exception("Non-MALA calculators cannot calculate ion-ion "
                            "energy yet.")


    def save_calculator(self, filename):
        """
        Saves enough information about the calculator to be reconstructed
        at a later time.

        TODO: Save some Espresso info here.

        Parameters
        ----------
        filename : string
            Path to file in which to store the Calculator.

        """
        if is_mala_available and self.mala_params is not None:
            self.mala_params.save(filename)
