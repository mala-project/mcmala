is_mala_available = True
try:
    import mala
except ModuleNotFoundError:
    is_mala_available = False
from ase.calculators.espresso import Espresso

class EspressoMC(Espresso):
    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='espresso', atoms=None, mala_params=None, **kwargs):
        super(EspressoMC, self).__init__(restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.observables_calculator = None
        self.mala_params = mala_params
        if is_mala_available:
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
        if is_mala_available and mala_params is not None:
            mala_params.save(filename)
