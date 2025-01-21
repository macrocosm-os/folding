from abc import ABC, abstractmethod
import time
from typing import Tuple
import functools
import openmm as mm
from openmm import app
from openmm import unit

import bittensor as bt
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.logger import logger


class GenericSimulation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_simulation(self):
        pass

    @staticmethod
    def timeit(method):
        @functools.wraps(method)
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Method {method.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return timed


class OpenMMSimulation(GenericSimulation):
    def _prepare_simulation(
        self, pdb: app.PDBFile, system_config: dict, seed: int = None
    ) -> Tuple[mm.app.Simulation, mm.System, app.Modeller, mm.Platform, dict]:
        """Common setup logic for creating OpenMM simulations."""
        start_time = time.time()
        forcefield = app.ForceField(system_config["ff"], system_config["water"])
        logger.warning(f"Creating ff took {time.time() - start_time:.4f} seconds")

        modeller = app.Modeller(pdb.topology, pdb.positions)

        start_time = time.time()
        modeller.deleteWater()
        logger.warning(f"Deleting water took {time.time() - start_time:.4f} seconds")

        # modeller.addExtraParticles(forcefield)

        start_time = time.time()
        modeller.addHydrogens(forcefield)
        logger.warning(f"Adding hydrogens took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        # modeller.addSolvent(
        #     forcefield,
        #     padding=system_config.box_padding * unit.nanometer,
        #     boxShape=system_config.box,
        # )
        logger.warning(f"Adding solvent took {time.time() - start_time:.4f} seconds")

        # Create the system
        start_time = time.time()
        # The assumption here is that the system_config cutoff MUST be given in nanometers
        threshold = (pdb.topology.getUnitCellDimensions().min().value_in_unit(mm.unit.nanometers)) / 2
        if system_config["cutoff"] > threshold:
            nonbondedCutoff = threshold * mm.unit.nanometers
            # set the attribute in the config for the pipeline.
            system_config["cutoff"] = threshold
            logger.warning(
                f"Nonbonded cutoff is greater than half the minimum box dimension. Setting nonbonded cutoff to {threshold} nm"
            )
        else:
            nonbondedCutoff = system_config["cutoff"] * mm.unit.nanometers

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=mm.app.NoCutoff,
            nonbondedCutoff=nonbondedCutoff,
            constraints=system_config["constraints"],
        )
        logger.warning(f"Creating system took {time.time() - start_time:.4f} seconds")

        # Integrator settings
        integrator = mm.LangevinIntegrator(
            system_config["temperature"] * unit.kelvin,
            system_config["friction"] / unit.picosecond,
            system_config["time_step_size"] * unit.picoseconds,
        )

        seed = seed if seed is not None else system_config["seed"]
        integrator.setRandomNumberSeed(seed)

        # Periodic boundary conditions
        # pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

        # if state != "nvt":
        #     system.addForce(
        #         mm.MonteCarloBarostat(
        #             system_config["pressure"] * unit.bar,
        #             system_config["temperature"] * unit.kelvin,
        #         )
        #     )

        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {
            "DeterministicForces": "true",
            "Precision": "double",
            "DisablePmeStream": "true",  # Reference: https://github.com/openmm/openmm/issues/3589
        }

        simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)

        return simulation, system, modeller, platform, properties

    @GenericSimulation.timeit
    def create_simulation(
        self, pdb: app.PDBFile, system_config: dict, seed: int = None
    ) -> Tuple[app.Simulation, SimulationConfig]:
        """Creates a new simulation object with the provided parameters."""
        start_time = time.time()
        simulation, _, modeller, _, _ = self._prepare_simulation(pdb, system_config, seed)

        start_time = time.time()
        simulation.context.setPositions(modeller.positions)
        logger.warning(f"Setting positions took {time.time() - start_time:.4f} seconds")

        # Convert system_config values to appropriate types
        for k, v in system_config.items():
            if not isinstance(v, (str, int, float, dict)):
                system_config[k] = str(v)

        return simulation, SimulationConfig(**system_config)

    def create_simulation_from_checkpoint(
        self,
        pdb: app.PDBFile,
        system_config: dict,
        checkpoint_path: str,
        seed: int = None,
    ) -> Tuple[app.Simulation, SimulationConfig]:
        """Creates a simulation object from a checkpoint file."""
        temp_sim, system, modeller, platform, properties = self._prepare_simulation(pdb, system_config, seed)

        # Load checkpoint state
        temp_sim.loadCheckpoint(checkpoint_path)
        positions = temp_sim.context.getState(getPositions=True).getPositions()
        velocities = temp_sim.context.getState(getVelocities=True).getVelocities()
        step = temp_sim.currentStep
        
        integrator = mm.LangevinIntegrator(
            system_config["temperature"] * unit.kelvin,
            system_config["friction"] / unit.picosecond,
            system_config["time_step_size"] * unit.picoseconds,
        )

        seed = seed if seed is not None else system_config["seed"]
        integrator.setRandomNumberSeed(seed)

        clean_sim = app.Simulation(modeller.topology, system, integrator, platform, properties)
        clean_sim.context.setPositions(positions)
        clean_sim.context.setVelocities(velocities)
        clean_sim.currentStep = step

        return clean_sim, SimulationConfig(**system_config)
