from qepy.calculator import QEpyCalculator
from ase.calculators.calculator import all_changes
from ase.io.espresso import read_fortran_namelist
from ase.io import write, read
from mcmala.common.parallelizer import get_comm, get_rank
from mcmala.common.converters import kelvin_to_rydberg
is_mala_available = True
try:
    import mala
except ModuleNotFoundError:
    is_mala_available = False


class EspressoMC(QEpyCalculator):
    def __init__(self, temperature=None, atoms=None, task='scf', embed=None,
                 inputfile=None, input_data=None, wrap=False,
                 lmovecell=False, mala_params=None, temp_folder=None, **kwargs):
        """

        Parameters
        ----------
        restart
        ignore_bad_restart_file
        label
        atoms
        mala_params
        kwargs
        """
        # TODO: For now - make this pretty later. This will break super easy
        # and it is ugly
        if inputfile is not None:
            inputfile = self._copy_and_modify_input_file(inputfile, temperature,
                                                         temp_folder)

        super(EspressoMC, self).__init__(atoms=atoms, comm=get_comm(),
                                         task=task,
                                         embed=embed, inputfile=inputfile,
                                         input_data=input_data, wrap=wrap,
                                         lmovecell=lmovecell, **kwargs)

        self.observables_calculator = None
        self.mala_params = mala_params
        if is_mala_available and mala_params is not None:
            self.observables_calculator = mala.LDOS(mala_params)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        # TODO: Implement a check whether these atoms have already been
        # calculated. So far we recalculate every time.
        self.results["energy"] = self.get_potential_energy(atoms=atoms)

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

    def _copy_and_modify_input_file(self, filename, temperature,
                                    temp_folder):
        input_file = open(filename)
        atoms = read(filename)
        name_list = read_fortran_namelist(input_file)
        input_data = {**{**dict(name_list[0]["control"]),
                         **dict(name_list[0]["electrons"])},
                      **dict(name_list[0]["system"])}
        other_data_list = name_list[1]
        found_k_points = False
        found_atomic_species = False

        for entry in other_data_list:
            if found_k_points:
                k_points = (
                entry.split()[0], entry.split()[1], entry.split()[2])
                found_k_points = False

            if found_atomic_species:
                pseudopotentials = {entry.split()[0]: entry.split()[-1]}
                found_atomic_species = False

            if "K_POINTS" in entry:
                found_k_points = True

            if "ATOMIC_SPECIES" in entry:
                found_atomic_species = True

        input_data["outdir"] = temp_folder
        out_file_name = filename
        if temperature is not None:
            out_file_name = str(temperature) + "K" + filename
            input_data["degauss"] = round(kelvin_to_rydberg(temperature), 7)
        if get_rank() == 0:
            write(out_file_name, atoms, "espresso-in",
                  input_data=input_data, pseudopotentials=pseudopotentials,
                  kpts=k_points, parallel=False)
        return out_file_name
