from abc import ABC, abstractmethod
import time
from typing import Tuple
import functools
import openmm as mm
from openmm import app
from openmm import unit


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
            logger.info(
                f"Method {method.__name__} took {end_time - start_time:.4f} seconds"
            )
            return result

        return timed


class OpenMMSimulation(GenericSimulation):
    def __init__(
        self, default_simulation_properties: dict = None, verbose: bool = False
    ):
        """Initialize the OpenMMSimulation object.

        Args:
            default_simulation_properties (dict, optional): A dictionary of default simulation properties. Defaults to None.
            verbose (bool, optional): A boolean flag to determine if the simulation should be verbose. Defaults to False.
        """
        self.setup_times = {}

        # Reference for DisablePmeStream: https://github.com/openmm/openmm/issues/3589
        self.default_simulation_properties = default_simulation_properties or {
            "DeterministicForces": "true",
            "Precision": "double",
            "DisablePmeStream": "true",
        }

        self.verbose = verbose

    @GenericSimulation.timeit
    def _setup_forcefield(self, ff: str, water: str):
        forcefield = app.ForceField(ff, water)
        return forcefield

    @GenericSimulation.timeit
    def _setup_modeller(self, pdb: app.PDBFile):
        modeller = app.Modeller(pdb.topology, pdb.positions)
        return modeller

    @GenericSimulation.timeit
    def _initialize_fluid(
        self, modeller: app.Modeller, forcefield: app.ForceField
    ) -> app.Modeller:
        modeller.deleteWater()
        modeller.addHydrogens(forcefield)

        return modeller

    @GenericSimulation.timeit
    def _use_solvent(
        self,
        modeller: app.Modeller,
        forcefield: app.ForceField,
        box_padding: float,
        box_shape: str,
    ) -> app.Modeller:
        modeller.addSolvent(
            forcefield,
            padding=box_padding * unit.nanometer,
            boxShape=box_shape,
        )

        return modeller

    @GenericSimulation.timeit
    def _add_extra_particles(
        self, modeller: app.Modeller, forcefield: app.ForceField
    ) -> app.Modeller:
        modeller.addExtraParticles(forcefield)
        return modeller

    @GenericSimulation.timeit
    def _create_system(
        self,
        modeller: app.Modeller,
        forcefield: app.ForceField,
        cutoff: float,
        constraints: str,
    ) -> Tuple[mm.System, float]:
        threshold = (
            modeller.topology.getUnitCellDimensions()
            .min()
            .value_in_unit(mm.unit.nanometers)
        ) / 2

        nonbondedCutoff = min(cutoff, threshold) * mm.unit.nanometers
        if cutoff > threshold:
            logger.debug(
                f"Nonbonded cutoff is greater than half the minimum box dimension. Setting nonbonded cutoff to {threshold} nm"
            )

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=mm.app.NoCutoff,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
        )
        return system, cutoff

    @GenericSimulation.timeit
    def _setup_integrator(
        self, temperature: float, friction: float, time_step_size: float, seed: int
    ) -> mm.LangevinIntegrator:
        integrator = mm.LangevinIntegrator(
            temperature * unit.kelvin,
            friction / unit.picosecond,
            time_step_size * unit.picoseconds,
        )

        integrator.setRandomNumberSeed(seed)
        return integrator

    @GenericSimulation.timeit
    def _setup_simulation(
        self,
        modeller: app.Modeller,
        system: mm.System,
        integrator: mm.LangevinIntegrator,
        properties: dict,
    ) -> app.Simulation:
        platform = mm.Platform.getPlatformByName("CUDA")

        simulation = mm.app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )

        # Set initial positions
        simulation.context.setPositions(modeller.positions)
        return simulation

    @GenericSimulation.timeit
    def pipeline(
        self,
        pdb: app.PDBFile,
        use_solvent: bool,
        include_fluid: bool,
        system_config: dict,
        simulation_properties: dict,
        seed: int,
    ) -> Tuple[app.Simulation, app.Modeller, mm.LangevinIntegrator, dict]:
        """Creates a simulation object with the given parameters.

        Args:
            pdb (app.PDBFile): The PDB file to use for the simulation.
            use_solvent (bool): Whether to use solvent for the simulation.
            system_config (dict): The system configuration to use for the simulation.
            simulation_properties (dict): The simulation properties to use for the simulation.
            seed (int): The seed to use for the simulation.

        Returns:
            Tuple[app.Simulation, app.Modeller, mm.LangevinIntegrator, dict]:
                A tuple containing the simulation object, the modeller object, the integrator object, and the system configuration.
        """
        forcefield = self._setup_forcefield(
            ff=system_config["ff"], water=system_config["water"]
        )
        modeller = self._setup_modeller(pdb=pdb)

        if use_solvent or include_fluid:
            modeller = self._initialize_fluid(modeller=modeller, forcefield=forcefield)

        if use_solvent:
            modeller = self._use_solvent(
                modeller=modeller,
                forcefield=forcefield,
                box_padding=system_config["box_padding"],
                box_shape=system_config["box"],
            )

            modeller = self._add_extra_particles(
                modeller=modeller, forcefield=forcefield
            )

        system, cutoff = self._create_system(
            modeller=modeller,
            forcefield=forcefield,
            cutoff=system_config["cutoff"],
            constraints=system_config["constraints"],
        )
        # could change in the process of creating the system
        system_config["cutoff"] = cutoff

        integrator = self._setup_integrator(
            temperature=system_config["temperature"],
            friction=system_config["friction"],
            time_step_size=system_config["time_step_size"],
            seed=seed,
        )

        simulation = self._setup_simulation(
            modeller=modeller,
            system=system,
            integrator=integrator,
            properties=simulation_properties,
        )

        return simulation, system_config

    @GenericSimulation.timeit
    def from_pipeline(
        self,
        pdb: app.PDBFile,
        system_config: dict,
        simulation_properties: dict,
        seed: int,
    ) -> Tuple[app.Simulation, app.Modeller, mm.LangevinIntegrator, dict]:
        """Creates a simulation object from the given parameters.

        Args:
            pdb (app.PDBFile): The PDB file to use for the simulation.
            system_config (dict): The system configuration to use for the simulation.
            simulation_properties (dict): The simulation properties to use for the simulation.
            seed (int): The seed to use for the simulation.
        """
        return self.pipeline(
            pdb=pdb,
            include_fluid=True,
            use_solvent=False,
            system_config=system_config,
            simulation_properties=simulation_properties,
            seed=seed,
        )

    @GenericSimulation.timeit
    def from_solvent_pipeline(
        self,
        pdb: app.PDBFile,
        system_config: dict,
        simulation_properties: dict,
        seed: int,
    ) -> Tuple[app.Simulation, app.Modeller, mm.LangevinIntegrator, dict]:
        """Creates a simulation object from the given parameters.

        Importantly, when the validator creates a simulation with solvent involved,
        the miner instantiation pipeline becomes more complicated due to the presence of fluids.

        Therefore, if a pdb has been initialized with solvent from the validator, we must
        skip all the steps that contain fluid, and NOT initialize the simulation with solvent information.

        Args:
            pdb (app.PDBFile): The PDB file to use for the simulation.
            system_config (dict): The system configuration to use for the simulation.
            simulation_properties (dict): The simulation properties to use for the simulation.
            seed (int): The seed to use for the simulation.

        Returns:
            Tuple[app.Simulation, app.Modeller, mm.LangevinIntegrator, dict]:
                A tuple containing the simulation object, the modeller object, the integrator object, and the system configuration.
        """
        return self.pipeline(
            pdb=pdb,
            include_fluid=False,
            use_solvent=False,
            system_config=system_config,
            simulation_properties=simulation_properties,
            seed=seed,
        )

    @GenericSimulation.timeit
    def create_simulation(
        self,
        pdb: app.PDBFile,
        with_solvent: bool,
        system_config: dict,
        seed: int = None,
    ) -> Tuple[app.Simulation, SimulationConfig]:
        seed = seed if seed is not None else system_config["seed"]

        simulation, system_config = self.pipeline(
            pdb=pdb,
            include_fluid=True,
            use_solvent=with_solvent,
            system_config=system_config,
            simulation_properties=self.default_simulation_properties,
            seed=seed,
        )

        # Converting the system config into a Dict[str,str] and ensure all values in system_config are of the correct type
        for k, v in system_config.items():
            if not isinstance(v, (str, int, float, dict)):
                system_config[k] = str(v)

        return simulation, SimulationConfig(**system_config)
