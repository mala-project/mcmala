from ase.calculators.calculator import all_changes
from ase.io import write, read
from mcmala.common.parallelizer import get_comm, get_rank, printout
from mcmala.common.converters import kelvin_to_rydberg
from os.path import join

from mcmala.simulation.available_frameworks import is_qepy_available, \
    is_mala_available

if is_mala_available:
    import mala

if is_qepy_available:
    from qepy.calculator import QEpyCalculator

if is_qepy_available:
    class EspressoMC(QEpyCalculator):
        def __init__(self, atoms, input_data,
                     pseudopotentials, kpts, temperature=None,
                     mala_params=None, working_directory="./",
                     input_file_name="default.pwi",
                     **kwargs):
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
            # Becomes True once the input file has ben created.
            self._is_initialized = False
            self.is_initialized = False

            super(EspressoMC, self).__init__(atoms=atoms, comm=get_comm(),
                                             input_data=input_data, **kwargs)

            self.observables_calculator = None
            self.mala_params = mala_params
            if is_mala_available and mala_params is not None:
                self.observables_calculator = mala.LDOS(mala_params)

            # Atoms are copied in the constructor of the
            # parent class, the rest here.
            self.kpts = kpts
            self.pseudopotentials = pseudopotentials
            self.input_data = input_data

            # These are rarely given upon creation time.
            self._working_directory = "./"
            self._input_file_name = "None"

            self.working_directory = working_directory
            self.input_file_name = input_file_name
            self.temperature = temperature

        @property
        def temperature(self):
            return self._temperature

        @temperature.setter
        def temperature(self, value):
            if not self.is_initialized:
                self._temperature = value
            else:
                printout("Simulation already initialized, cannot change "
                         "the temperature now.")

        @property
        def working_directory(self):
            return self._working_directory

        @working_directory.setter
        def working_directory(self, value):
            if not self.is_initialized:
                self._working_directory = value
                self.inputfile = join(value, self.input_file_name)
            else:
                printout("Simulation already initialized, cannot change "
                         "the working directory now.")

        @property
        def input_file_name(self):
            return self._input_file_name

        @input_file_name.setter
        def input_file_name(self, value):
            if not self.is_initialized:
                self._input_file_name = value
                self.inputfile = join(self.working_directory, value)
            else:
                printout("Simulation already initialized, cannot change "
                         "the input file name now.")

        @property
        def is_initialized(self):
            return self._is_initialized

        @is_initialized.setter
        def is_initialized(self, value):
            if self._is_initialized is True:
                return
            else:
                self._is_initialized = value

        def calculate(self, atoms=None, properties=['energy'],
                      system_changes=all_changes):
            # TODO: Implement a check whether these atoms have already been
            # calculated. So far we recalculate every time.
            if not self.is_initialized:
                self.create_input_file()
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
                self.mala_params.save(filename+".params.json")

        def update_paths(self, working_directory, markov_chain_id):
            """
            Update the working path(s) used by this calculator.

            Parameters
            ----------
            working_directory : string
                Working directory of the Markov chain.

            markov_chain_id : string
                ID under which the Markov chain that calls this calculator
                operates.
            """
            self.working_directory = join(working_directory, markov_chain_id)
            self.input_file_name = markov_chain_id + "_espresso.pwi"

        def create_input_file(self):
            self.input_data["outdir"] = join(self.working_directory, "temp")
            if self.temperature is not None:
                self.input_data["degauss"] = \
                    round(kelvin_to_rydberg(self.temperature), 7)
            if get_rank() == 0:
                write(self.inputfile, self.atoms, "espresso-in",
                      input_data=self.input_data,
                      pseudopotentials=self.pseudopotentials,
                      kpts=self.kpts, parallel=False)
            self.is_initialized = True

        @classmethod
        def from_input_file(cls, working_directory, input_file_name):
            path_to_file = join(working_directory, input_file_name)
            atoms = read(path_to_file)
            new_calculator = EspressoMC(atoms, None, None, None,
                                        working_directory=working_directory,
                                        input_file_name=input_file_name)
            new_calculator.is_initialized = True
            return new_calculator
