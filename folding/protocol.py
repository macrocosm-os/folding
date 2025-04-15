# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Macrocosmos AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import base64
import bittensor as bt
from folding.utils.logger import logger


class PingSynapse(bt.Synapse):
    """Responsible for determining if a miner can accept a request"""

    can_serve: bool = False
    available_compute: typing.Optional[int] = None  # TODO: number of threads / gpus?


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
        # Right here we perform validation that the response has expected hash
        if not isinstance(self.md_output, dict):
            self.md_output = {}
        else:
            md_output = {}
            # Access fields directly from the MDOutput model
            for field in self.md_output.keys():
                value = self.md_output[field]
                if value is not None:
                    try:
                        md_output[field] = base64.b64decode(value)
                    except Exception as e:
                        logger.error(f"Error decoding {field} from md_output: {e}")
                        md_output[field] = None
                else:
                    md_output[field] = None

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


class IntermediateSubmissionSynapse(bt.Synapse):
    """A synapse for submission of intermediate checkpoints.

    Attributes:
    - pdb_id: A Protein id
    - job_id: A job id to retrieve the job from the GJP.
    - checkpoint_numbers: A list of checkpoints to submit.
    """

    pdb_id: str
    job_id: str
    checkpoint_numbers: list[int]

    # Optional request output, filled by receiving axon.
    cpt_files: typing.Optional[dict] = None

    def deserialize(self) -> int:
        """
        Deserialize the output. This method retrieves the response from
        the miner in the form of a bytestream, deserializes it and returns it
        as the output of the dendrite.query() call.
        """
        if not isinstance(self.cpt_files, dict):
            self.cpt_files = {}
        else:
            cpt_files = {}
            for k, v in self.cpt_files.items():
                if v is not None:
                    try:
                        cpt_files[k] = base64.b64decode(v)
                    except Exception as e:
                        logger.error(f"Error decoding {k} from cpt_files: {e}")
                        cpt_files[k] = None
                else:
                    cpt_files[k] = None

            self.cpt_files = cpt_files

        return self


class DFTJobSubmissionSynapse(bt.Synapse):
    """A synapse for submission of DFT jobs."""

    job_id: str
    geometry: str

    def deserialize(self) -> int:
        return self
