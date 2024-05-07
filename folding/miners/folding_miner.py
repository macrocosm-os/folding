import os
import time
import glob
import base64
import psutil
import concurrent.futures
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass

import bittensor as bt

# import base miner class which takes care of most of the boilerplate
from folding.base.miner import BaseMinerNeuron
from folding.protocol import FoldingSynapse
from folding.utils.ops import run_cmd_commands, check_if_directory_exists

# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")


def attach_files_to_synapse(
    synapse: FoldingSynapse,
    data_directory: str,
    state: str,
) -> FoldingSynapse:
    """load the output files as bytes and add to synapse.md_output

    Args:
        synapse (FoldingSynapse): Recently received synapse object
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
        3. .edr (created during mdrun commands, logged every nstenergy steps)
        4. .cpt (created during mdrun commands, logged every nstcheckpoint steps)


    Returns:
        FoldingSynapse: synapse with md_output attached
    """

    synapse.md_output = {}  # ensure that the initial state is empty

    try:
        state_files = os.path.join(
            data_directory, f"{state}"
        )  # mdrun commands make the filenames [state.*]

        # applying glob to state_files will get the necessary files we need (e.g. nvt.tpr, nvt.xtc, nvt.cpt, nvt.edr, etc.)
        all_state_files = glob.glob(f"{state_files}*")  # Grab all the state_files
        # bt.logging.info(f"all_state_files: {all_state_files}")
        latest_cpt_file = glob.glob("*.cpt")
        # bt.logging.info(f"latest_cpt_file: {latest_cpt_file}")

        files_to_attach: List = (
            all_state_files + latest_cpt_file
        )  # combine the state files and the latest checkpoint file

        bt.logging.info(f"Sending files to validator: {files_to_attach}")
        for filename in files_to_attach:
            bt.logging.info(f"Attaching file: {filename!r} to synapse.md_output")
            try:
                with open(filename, "rb") as f:
                    synapse.md_output[filename] = base64.b64encode(f.read())
            except Exception as e:
                bt.logging.error(f"Failed to read file {filename!r} with error: {e}")

        bt.logging.success(
            f"✅ Attached {len(synapse.md_output)} files to synapse.md_output for protein: {synapse.pdb_id} ✅"
        )

    except Exception as e:
        bt.logging.error(f"Failed to attach files with error: {e}")

    finally:
        return synapse  # either return the synapse wth the md_output attached or the synapse as is.


def create_empty_file(file_path: str):
    with open(file_path, "w") as f:
        pass


class FoldingMiner(BaseMinerNeuron):
    def __init__(self, config=None, base_data_path: str = None):
        super().__init__(config=config)

        # TODO: There needs to be a timeout manager. Right now, if
        # the simulation times out, the only time the memory is freed is when the miner
        # is restarted, or sampled again.

        def nested_dict():
            return defaultdict(
                lambda: None
            )  # allows us to set the desired attribute to anything.

        self.base_data_path = (
            base_data_path if base_data_path is not None else BASE_DATA_PATH
        )
        self.simulations = defaultdict(
            nested_dict
        )  # Maps pdb_ids to the current state of the simulation

        self.max_num_processes = psutil.cpu_count(logical=False)  # Only physical cores
        self.max_workers = (
            self.max_num_processes - 1
        )  # subtract one to ensure that we are not using all the processors.

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )  # remove one for safety

    def configure_commands(self, mdrun_args: str) -> Dict[str, List[str]]:
        commands = [
            "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",  # Temperature equilibration
            "gmx mdrun -deffnm nvt " + mdrun_args,
            "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",  # Pressure equilibration
            "gmx mdrun -deffnm npt " + mdrun_args,
            f"gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr",  # Production run
            f"gmx mdrun -deffnm md_0_1 " + mdrun_args,
            f"echo '1\n1\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_center.xtc -center -pbc mol",
        ]

        # These are rough identifiers for the different states of the simulation
        state_commands = {
            "nvt": commands[:2],
            "npt": commands[2:4],
            "md_0_1": commands[4:],
        }

        return state_commands

    def forward(self, synapse: FoldingSynapse) -> FoldingSynapse:
        """
        The main async function that is called by the dendrite to run the simulation.
        There are a set of default behaviours the miner should carry out based on the form the synapse comes in as:

            1. If the synapse has a pdb_id that is already being processed, return the intermediate gro file
            2. If the synapse md_inputs contains a ckpt file, then we are expected to either accept/reject a simulation rebase.
            3. If none of the above conditions are met, we start a new simulation.
                - If the number of active processes is less than the number of CPUs and the pdb_id is unique, start a new process

        Returns:
            FoldingSynapse: synapse with md_output attached
        """

        # If we are already running a process with the same identifier, return intermediate information
        bt.logging.info(f"⌛ Query from validator for protein: {synapse.pdb_id} ⌛")
        if synapse.pdb_id in self.simulations:
            simulation = self.simulations[synapse.pdb_id]
            current_executor_state = simulation["executor"].get_state()

            if current_executor_state == "finished":
                final_synapse = attach_files_to_synapse(
                    synapse=synapse,
                    data_directory=simulation["output_dir"],
                    state="md_0_1",  # attach the last state of the simulation as not files are labelled as 'finished'
                )

                # This will run if final_synapse exists.
                del self.simulations[
                    synapse.pdb_id
                ]  # Remove the simulation from the list

                # TODO: Here, place some type of delete method to remove some files?
                return final_synapse

            else:
                # Don't delete the simulation if it's not finished
                return attach_files_to_synapse(
                    synapse=synapse,
                    data_directory=simulation["output_dir"],
                    state=current_executor_state,
                )

        if os.path.exists(self.base_data_path) and synapse.pdb_id in os.listdir(
            self.base_data_path
        ):
            # If we have a pdb_id in the data directory, we can assume that the simulation has been run before
            # and we can return the files from the last simulation. This only works if you have kept the data.
            output_dir = os.path.join(self.base_data_path, synapse.pdb_id)

            return attach_files_to_synapse(
                synapse=synapse, data_directory=output_dir, state="md_0_1"
            )

        # Check if the number of active processes is less than the number of CPUs
        if len(self.simulations) >= self.max_workers:
            bt.logging.warning("Cannot start new process: CPU limit reached.")
            return synapse  # Return the synapse as is

        state_commands = self.configure_commands(mdrun_args=synapse.mdrun_args)

        # Create the job and submit it to the executor
        simulation_manager = SimulationManager(
            pdb_id=synapse.pdb_id,
            output_dir=os.path.join(self.base_data_path, synapse.pdb_id),
        )

        future = self.executor.submit(
            simulation_manager.run,
            synapse.md_inputs,
            state_commands,
            self.config.neuron.suppress_cmd_output,
            self.config.mock,
        )

        self.simulations[synapse.pdb_id]["executor"] = simulation_manager
        self.simulations[synapse.pdb_id]["future"] = future
        self.simulations[synapse.pdb_id]["output_dir"] = simulation_manager.output_dir


class SimulationManager:
    def __init__(self, pdb_id: str, output_dir: str) -> None:
        self.pdb_id = pdb_id
        self.state: str = None
        self.state_file_name = f"{pdb_id}_state.txt"

        self.output_dir = output_dir

    def run(
        self,
        md_inputs: Dict,
        commands: Dict,
        suppress_cmd_output: bool = True,
        mock: bool = False,
    ):
        """run method to handle the processing of generic simulations.

        Args:
            synapse (FoldingSynapse): synapse object that contains the information for the simulation
            commands (Dict): Dict of lists of commands that we are meant to run in the executor
            config (Dict): configuration file for the miner
        """
        bt.logging.info(
            f"Running simulation for protein: {self.pdb_id} with files {md_inputs.keys()}"
        )

        start_time = time.time()

        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir)  # TODO: will this be a problem with many processes?

        # The following files are required for GROMACS simulations and are recieved from the validator
        for filename, content in md_inputs.items():
            # Write the file to the output directory
            with open(filename, "w") as file:
                bt.logging.info(f"\nWriting {filename} to {self.output_dir}")
                file.write(content)

        for state, commands in commands.items():
            bt.logging.info(f"Running {state} commands")
            with open(self.state_file_name, "w") as f:
                f.write(f"{state}\n")

            run_cmd_commands(commands=commands, suppress_cmd_output=suppress_cmd_output)

            if mock:
                bt.logging.warning("Running in mock mode, creating fake files...")
                for ext in ["tpr", "xtc", "edr", "cpt"]:
                    create_empty_file(os.path.join(self.output_dir, f"{state}.{ext}"))

        bt.logging.success(f"✅Finished simulation for protein: {self.pdb_id}✅")

        state = "finished"
        with open(self.state_file_name, "w") as f:
            f.write(f"{state}\n")

        total_run_rime = time.time() - start_time

    def get_state(self) -> str:
        """get_state reads a txt file that contains the current state of the simulation"""
        with open(os.path.join(self.output_dir, self.state_file_name), "r") as f:
            lines = f.readlines()

            if lines:
                return lines[-1].strip()  # return the last line of the file
            return None


class MockSimulationManager(SimulationManager):
    def __init__(self, pdb_id: str, output_dir: str) -> None:
        super().__init__(pdb_id=pdb_id)
        self.required_values = set(["init", "wait", "finished"])
        self.output_dir = output_dir

    def run(self, total_wait_time: int = 1):
        start_time = time.time()

        bt.logging.success(f"✅ MockSimulationManager.run is running ✅")
        check_if_directory_exists(output_directory=self.output_dir)

        store = os.path.join(self.output_dir, self.state_file_name)
        states = ["init", "wait", "finished"]

        itermediate_interval = total_wait_time / len(states)

        for state in states:
            bt.logging.info(f"Running state: {state}")
            state_time = time.time()
            with open(store, "w") as f:
                f.write(f"{state}\n")

            time.sleep(itermediate_interval)
            bt.logging.info(f"Total state_time: {time.time() - state_time}")

        bt.logging.warning(f"Total run method time: {time.time() - start_time}")
