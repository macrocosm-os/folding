import typing
import base64
import bittensor as bt
from folding.utils.logger import logger


class PingSynapse(bt.Synapse):
    """Responsible for determining if a miner can accept a request"""

    can_serve: bool = False
    available_compute: typing.Optional[int] = None  # TODO: number of threads / gpus?


class ParticipationSynapse(bt.Synapse):
    """Responsible for determining if a miner is participating in a specific job"""

    job_id: str
    is_participating: bool = False


class JobSubmissionSynapse(bt.Synapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - pdb_id: A Protein id, which contains the necessary details of the protein to be folded.
    - md_inputs: A dictionary containing the input files for the openmm simulation.
    - system_config: A dictionary containing the system configuration for the simulation.
    - md_output: A dictionary containing the output files of the openmm simulation.
    - miner_seed: An integer value which is the seed for the simulation.
    - miner_state: A string value which is the state of the miner.
    """

    pdb_id: str
    job_id: str

    best_submitted_energy: typing.Optional[float] = None

    # Optional request output, filled by receiving axon.
    md_output: typing.Optional[dict] = None
    miner_seed: typing.Optional[int] = None
    miner_state: typing.Optional[str] = None

    def deserialize(self) -> int:
        """
        Deserialize the output. This method retrieves the response from
        the miner in the form of a bytestream, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - dict: The serialized response, which in this case is the value of md_output.
        """
        logger.info(
            f"Deserializing response from miner, I am: {self.pdb_id}, hotkey: {self.axon.hotkey[:8]}"
        )
        # Right here we perform validation that the response has expected hash
        if not isinstance(self.md_output, dict):
            self.md_output = {}
        else:
            md_output = {}
            for k, v in self.md_output.items():
                try:
                    md_output[k] = base64.b64decode(v)
                except Exception as e:
                    logger.error(f"Error decoding {k} from md_output: {e}")
                    md_output[k] = None

            self.md_output = md_output

        return self


class OrganicSynapse(bt.Synapse):
    """A synapse for organic scoring."""

    pdb_id: str
    source: str
    ff: str
    water: str
    box: str
    temperature: float
    friction: float
    epsilon: float

    is_processed: typing.Optional[bool] = False

    def deserialize(self) -> dict:
        return self.dict()

    def get_simulation_params(self):
        return {
            "pdb_id": self.pdb_id,
            "source": self.source,
            "ff": self.ff,
            "water": self.water,
            "box": self.box,
            "temperature": self.temperature,
            "friction": self.friction,
            "epsilon": self.epsilon,
        }
