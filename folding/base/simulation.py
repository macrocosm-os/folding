from abc import ABC, abstractmethod
import openmm as mm
from openmm import app
from openmm import unit


class GenericSimulation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_simulation(self):
        pass


class OpenMMSimulation(GenericSimulation):
    def create_simulation(
        self, pdb: app.PDBFile, system_config, seed: str, state: str
    ) -> app.Simulation:
        """Recreates a simulation object based on the provided parameters.

        This method takes in a seed, state, and checkpoint file path to recreate a simulation object.
        Args:
            seed (str): The seed for the random number generator.
            state (str): The state of the simulation.
            cpt_file (str): The path to the checkpoint file.

        Returns:
            app.Simulation: The recreated simulation object.
        """
        forcefield = app.ForceField(system_config.ff, system_config.water)

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.deleteWater()

        modeller.addSolvent(
            forcefield,
            padding=system_config.box_padding * unit.nanometer,
            boxShape=system_config.box,
            model=system_config.water,
        )

        # Create the system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=system_config.nonbonded_method,
            nonbondedCutoff=system_config.cutoff * mm.unit.nanometers,
            constraints=system_config.constraints,
        )

        # Integrator settings
        integrator = mm.LangevinIntegrator(
            system_config.temperature * unit.kelvin,
            system_config.friction / unit.picosecond,
            system_config.time_step_size * unit.picoseconds,
        )
        integrator.setRandomNumberSeed(seed)
        # Periodic boundary conditions
        pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

        if state != "nvt":
            system.addForce(
                mm.MonteCarloBarostat(
                    system_config.pressure * unit.bar,
                    system_config.temperature * unit.kelvin,
                )
            )
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"DeterministicForces": "true", "Precision": "double"}
        simulation = mm.app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        # Create the simulation object
        return simulation
