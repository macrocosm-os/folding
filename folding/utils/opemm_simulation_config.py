from typing import Literal, Optional
from openmm import app
import openmm as mm
from enum import Enum
from pydantic import BaseModel


class NonbondedMethod(Enum):
    PME = mm.NonbondedForce.PME
    NoCutoff = mm.NonbondedForce.NoCutoff


class Constraints(Enum):
    Unconstricted = None
    HBonds = app.HBonds
    AllBonds = app.AllBonds
    HAngles = app.HAngles


class SimulationConfig(BaseModel):
    ff: str
    water: str
    box: Literal["cube", "dodecahedron", "octahedron"]
    temperature: float = 300.0
    time_step_size: float = 0.002
    time_units: str = "picosecond"
    save_interval_checkpoint: int = 5000
    save_interval_log: int = 100
    box_padding: float = 1.0
    friction: float = 1.0
    nonbonded_method: NonbondedMethod = NonbondedMethod.PME
    constraints: Constraints = Constraints.HBonds
    cutoff: Optional[float] = 1.0
    pressure: float = 1.0
    max_steps_nvt: int = 50000

    emtol: float = 1000.0
    nsteps_minimize: int = 100
    rvdw: float = 1.2
    rcoulomb: float = 1.2
    fourier_spacing: float = 0.15
    tau_p: float = 5.0
    ref_p: float = 1.0
    ref_t: float = 300.0
    gen_temp: float = 300.0

    # TODO: Do we need this?
    def apply_input(self):
        class Config:
            arbitrary_types_allowed = True
