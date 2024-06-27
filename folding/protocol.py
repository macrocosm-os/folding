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
import os
import glob

import bittensor as bt

from folding.utils.ops import get_tracebacks


class FoldingSynapse(bt.Synapse):
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
        if type(self.md_output) != dict:
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


class GetFoldingSynapse(FoldingSynapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    The purpose of the GetFoldingSynapse will be to send data from the miner to the validator.

    """

    md_output: dict
    serves_job: bool

    def attach_files(self, files_to_attach: typing.List):
        """function that parses a list of files and attaches them to the synapse object"""
        bt.logging.info(f"Sending files to validator: {files_to_attach}")
        for filename in files_to_attach:
            # trrs are large, and validators don't need them.
            if filename.endswith(".trr"):
                continue

            try:
                with open(filename, "rb") as f:
                    filename = filename.split("/")[
                        -1
                    ]  # remove the directory from the filename
                    self.md_output[filename] = base64.b64encode(f.read())
            except Exception as e:
                bt.logging.error(f"Failed to read file {filename!r} with error: {e}")
                get_tracebacks()

    def attach_files_to_synapse(
        self,
        data_directory: str,
        state: str,
    ) -> FoldingSynapse:
        """load the output files as bytes and add to synapse.md_output

        Args:
            data_directory (str): directory where the miner is holding the necessary data for the validator.
            state (str): the current state of the simulation

        state is either:
        1. nvt
        2. npt
        3. md_0_1
        4. finished

        State depends on the current state of the simulation (controlled in GromacsExecutor.run() method).

        During the simulation procedure, the validator queries the miner for the current state of the simulation.
        The files that the miner needs to return are:
            1. .tpr (created during grompp commands)
            2. .xtc (created during mdrun commands, logged every nstxout-compressed steps)
            3. .cpt (created during mdrun commands, logged every nstcheckpoint steps) # TODO: remove (re create .gro file from .tpr and .xtc)


        """

        self.md_output = {}  # ensure that the initial state is empty

        try:
            state_files = os.path.join(
                data_directory, f"{state}"
            )  # mdrun commands make the filenames [state.*]

            # applying glob to state_files will get the necessary files we need (e.g. nvt.tpr, nvt.xtc, nvt.cpt, nvt.edr, etc.)
            all_state_files = glob.glob(f"{state_files}*")  # Grab all the state_files
            latest_cpt_file = glob.glob("*.cpt")

            files_to_attach: typing.List = (
                all_state_files + latest_cpt_file
            )  # combine the state files and the latest checkpoint file

            if len(files_to_attach) == 0:
                raise FileNotFoundError(
                    f"No files found for {state}"
                )  # if this happens, goes to except block

            self.attach_files(files_to_attach=files_to_attach)

        except Exception as e:
            bt.logging.error(
                f"Failed to attach files for pdb {self.pdb_id} with error: {e}"
            )
            get_tracebacks()
            self.md_output = {}
            # TODO Maybe in this point in the logic it makes sense to try and restart the sim.


class PostFoldingSynapse(FoldingSynapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    The purpose of the PostFoldingSynapse will be to send the required data from the validator to miners

    Attributes:
    - pdb_id: A Protein id, which contains the necessary details of the protein to be folded.
    - md_inputs: A dictionary containing the input files for the gromacs simulation.
    - mdrun_args: A string containing the arguments to be passed to the gromacs mdrun command.
    """

    # Required request input, filled by sending dendrite caller.
    md_inputs: dict

    # Optional runtime args for gromacs
    mdrun_args: str = ""
    job_cancelled: bool = False
