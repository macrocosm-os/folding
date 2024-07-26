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


class PingSynapse(bt.Synapse):
    """ Responsible for determining if a miner can accept a request """
    can_serve: bool = False
    available_compute: typing.Optional[int] = None #TODO: number of threads / gpus? 
    

class JobSubmissionSynapse(bt.Synapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - pdb_id: A Protein id, which contains the necessary details of the protein to be folded.
    - md_inputs: A dictionary containing the input files for the gromacs simulation.
    - mdrun_args: A string containing the arguments to be passed to the gromacs mdrun command.
    """

    # Required request input, filled by sending dendrite caller.
    pdb_id: str
    md_inputs: dict

    # Optional runtime args for gromacs
    mdrun_args: str = ""
    
    # Miner can decide if they are serving the request or not.
    miner_serving: bool = True

    # Optional request output, filled by recieving axon.
    md_output: typing.Optional[dict] = None

    def deserialize(self) -> int:
        """
        Deserialize the output. This method retrieves the response from
        the miner in the form of a bytestream, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - dict: The serialized response, which in this case is the value of md_output.
        """
        bt.logging.info(
            f"Deserializing response from miner, I am: {self.pdb_id}, hotkey: {self.axon.hotkey[:8]}"
        )
        # Right here we perform validation that the reponse has expected hash
        if not isinstance(self.md_output, dict):
            self.md_output = {}
        else:
            md_output = {}
            for k, v in self.md_output.items():
                try:
                    md_output[k] = base64.b64decode(v)
                except Exception as e:
                    bt.logging.error(f"Error decoding {k} from md_output: {e}")
                    md_output[k] = None

            self.md_output = md_output
        return self
