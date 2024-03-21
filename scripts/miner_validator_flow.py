import os
import tqdm
import glob
import base64
import bittensor as bt

import argparse
from folding.validators.protein import Protein

import typing
import random
import string

from concurrent.futures import ThreadPoolExecutor


# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_random_string(length=5):
    """Generate a random string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class MockFoldingSynapse:
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
    def __init__(
        self,
        pdb_id: str,
        md_inputs: dict = {},
        mdrun_args: str = "",
        md_output: typing.Optional[dict] = None,
    ) -> None:
        self.pdb_id = pdb_id
        self.md_inputs = md_inputs
        self.mdrun_args = mdrun_args
        self.md_output = md_output

    def deserialize(self) -> int:
        """
        Deserialize the output. This method retrieves the response from
        the miner in the form of a bytestream, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - dict: The serialized response, which in this case is the value of md_output.
        """
        bt.logging.info(f"Deserializing response from miner, I am: {self.pdb_id}")
        # Right here we perform validation that the reponse has expected hash
        if type(self.md_output) != dict:
            self.md_output = {}
        else:
            self.md_output = {k: base64.b64decode(v) for k, v in self.md_output.items()}
        return self


def validator_forward(pdb_id, ff="charmm27", box="dodecahedron", max_steps=10):
    protein = Protein(
        pdb_id=pdb_id,
        ff=ff,
        box=box,
        max_steps=max_steps,
    )
    bt.logging.info(f"Protein challenge: {protein}")

    synapse = MockFoldingSynapse(pdb_id=protein.pdb_id, md_inputs=protein.md_inputs)

    return synapse, protein


# This is the core miner function, which decides the miner's response to a valid, high-priority request.
def miner_forward(
    synapse: MockFoldingSynapse, hotkey, suppress_cmd_output=True
) -> MockFoldingSynapse:
    # This function runs after the synapse has been deserialized (i.e. after synapse.data is available).
    # This function runs after the blacklist and priority functions have been called.
    # TODO: Determine how many steps to run based on timeout
    bt.logging.info(
        f"Running GROMACS simulation for protein: {synapse.pdb_id} with files {synapse.md_inputs.keys()} mdrun_args: {synapse.mdrun_args}"
    )
    synapse.md_output = {}

    output_directory = os.path.join(
        ROOT_DIR, "folding", "data", synapse.pdb_id, "miners", hotkey
    )

    bt.logging.info(f"Attempting to make {output_directory} for miners!")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Change to output directory
    bt.logging.info(f"Changing directory to {output_directory}")
    os.chdir(output_directory)

    # The following files are required for GROMACS simulations and are recieved from the validator
    for filename, content in synapse.md_inputs.items():
        # Write the file to the output directory
        with open(filename, "w") as file:
            bt.logging.info(f"\nWriting {filename} to {output_directory}")
            file.write(content)

    # TODO(developer): Replace with actual implementation logic.
    commands = [
        "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",  # Temperature equilibration
        "gmx mdrun -deffnm nvt " + synapse.mdrun_args,
        "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",  # Pressure equilibration
        "gmx mdrun -deffnm npt " + synapse.mdrun_args,
        "gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr",  # Production run
        "gmx mdrun -deffnm md_0_1 " + synapse.mdrun_args,
    ]

    for cmd in tqdm.tqdm(commands):
        # We want to catch any errors that occur in the above steps and then return the error to the user
        bt.logging.info(f"Running GROMACS command: {cmd}")

        if suppress_cmd_output:
            cmd += " > /dev/null 2>&1"

        os.system(cmd)

    # load the output files as bytes and add to synapse.md_output
    for filename in glob.glob("md_0_1.*"):
        bt.logging.info(f"Attaching file: {filename!r} to synapse.md_output")
        try:
            with open(filename, "rb") as f:
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            bt.logging.error(f"Failed to read file {filename!r} with error: {e}")

    bt.logging.success(
        f"Finished running GROMACS simulation for protein: {synapse.pdb_id} and attached {len(synapse.md_output)} files to synapse.md_output"
    )

    return synapse


def run_miner_forward(protein, synapse, suppress_cmd_output, miner_id: int):
    hotkey = generate_random_string()  # The miner's hotkey

    # Call your function here
    bt.logging.info(f"Running miner {miner_id} forward....")
    synapse = miner_forward(
        synapse=synapse, hotkey=hotkey, suppress_cmd_output=suppress_cmd_output
    )
    bt.logging.info(f"Miner {miner_id} complete!")

    protein.save_md_outputs(
        md_output=synapse.md_output, hotkey=hotkey
    )  # This line saves the data to the pdb_id/dendrite


def main_process(args):
    bt.logging.info(f"About to run validator forward....")
    synapse, protein = validator_forward(pdb_id=args.pdb_id)
    bt.logging.success(f"Validator forward complete!")

    # Here will will run parallel processes to run all the "miners" at the same time.
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_miner_forward, protein, synapse, args.suppress_cmd_output, miner_id
            )
            for miner_id in range(args.num_miners)
        ]

        for future in futures:
            future.result()  # This blocks until the function is completed

    bt.logging.success(f"Miner forward complete!")

    # # apply save_md_outputs in the protein class so we can save the data.
    # protein.save_md_outputs(md_output=synapse.md_output, hotkey=hotkey) # This line saves the data to the pdb_id/dendrite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    parser.add_argument(
        "--pdb_id",
        type=str,
        default="5oxe",
        help="protein that you want to fold",
    )

    parser.add_argument(
        "--num_miners",
        help="Number of miner simulations to run.",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--suppress_cmd_output",
        action="store_false",
        default=True,
        help="suppress_cmd_output",
    )

    args = parser.parse_args()

    main_process(args=args)
