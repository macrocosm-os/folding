import os
import time
import glob
import base64
import random
import hashlib
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Tuple
import copy
import traceback

import bittensor as bt
import openmm as mm
import openmm.app as app

# import base miner class which takes care of most of the boilerplate
from folding.base.miner import BaseMinerNeuron
from folding.base.simulation import OpenMMSimulation
from folding.protocol import JobSubmissionSynapse
from folding.utils.logging import log_event
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter
from folding.utils.ops import (
    check_if_directory_exists,
    get_tracebacks,
    write_pkl,
)
from folding.utils.opemm_simulation_config import SimulationConfig

# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_PATH = os.path.join(ROOT_DIR, "miner-data")


def attach_files(
    files_to_attach: List, synapse: JobSubmissionSynapse
) -> JobSubmissionSynapse:
    """function that parses a list of files and attaches them to the synapse object"""
    bt.logging.info(f"Sending files to validator: {files_to_attach}")
    for filename in files_to_attach:
        try:
            with open(filename, "rb") as f:
                filename = filename.split("/")[
                    -1
                ]  # remove the directory from the filename
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            bt.logging.error(f"Failed to read file {filename!r} with error: {e}")
            get_tracebacks()

    return synapse


def attach_files_to_synapse(
    synapse: JobSubmissionSynapse,
    data_directory: str,
    state: str,
    seed: int,
) -> JobSubmissionSynapse:
    """load the output files as bytes and add to synapse.md_output

    Args:
        synapse (JobSubmissionSynapse): Recently received synapse object
        data_directory (str): directory where the miner is holding the necessary data for the validator.
        state (str): the current state of the simulation

    state is either:
     1. nvt
     2. npt
     3. md_0_1
     4. finished

    Returns:
        JobSubmissionSynapse: synapse with md_output attached
    """

    synapse.md_output = {}  # ensure that the initial state is empty

    try:
        state_files = os.path.join(data_directory, f"{state}")

        # This should be "state.cpt" and "state_old.cpt"
        all_state_files = glob.glob(f"{state_files}*")  # Grab all the state_files

        if len(all_state_files) == 0:
            raise FileNotFoundError(
                f"No files found for {state}"
            )  # if this happens, goes to except block

        synapse = attach_files(files_to_attach=all_state_files, synapse=synapse)

        synapse.miner_seed = seed
        synapse.miner_state = state

    except Exception as e:
        bt.logging.error(
            f"Failed to attach files for pdb {synapse.pdb_id} with error: {e}"
        )
        get_tracebacks()
        synapse.md_output = {}

    finally:
        return synapse  # either return the synapse wth the md_output attached or the synapse as is.


def check_synapse(
    self, synapse: JobSubmissionSynapse, event: Dict = None
) -> JobSubmissionSynapse:
    """Utility function to remove md_inputs if they exist"""
    if len(synapse.md_inputs) > 0:
        event["md_inputs_sizes"] = list(map(len, synapse.md_inputs.values()))
        event["md_inputs_filenames"] = list(synapse.md_inputs.keys())
        synapse.md_inputs = {}  # remove from synapse

    if synapse.md_output is not None:
        event["md_output_sizes"] = list(map(len, synapse.md_output.values()))
        event["md_output_filenames"] = list(synapse.md_output.keys())

    event["query_forward_time"] = time.time() - self.query_start_time

    log_event(self=self, event=event)
    return synapse


class FoldingMiner(BaseMinerNeuron):
    def __init__(self, config=None, base_data_path: str = None):
        super().__init__(config=config)

        # TODO: There needs to be a timeout manager. Right now, if
        # the simulation times out, the only time the memory is freed is when the miner
        # is restarted, or sampled again.

        self.base_data_path = (
            base_data_path
            if base_data_path is not None
            else os.path.join(BASE_DATA_PATH, self.wallet.hotkey.ss58_address[:8])
        )
        self.simulations = self.create_default_dict()

        self.max_workers = self.config.neuron.max_workers
        bt.logging.info(
            f"🚀 Starting FoldingMiner that handles {self.max_workers} workers 🚀"
        )

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )  # remove one for safety

        self.mock = None
        self.generate_random_seed = lambda: random.randint(0, 1000)

        # hardcorded for now -- TODO: make this more flexible
        self.STATES = ["nvt", "npt", "md_0_1"]
        self.CHECKPOINT_INTERVAL = 10000
        self.STATE_DATA_REPORTER_INTERVAL = 10
        self.EXIT_REPORTER_INTERVAL = 10

    def create_default_dict(self):
        def nested_dict():
            return defaultdict(
                lambda: None
            )  # allows us to set the desired attribute to anything.

        return defaultdict(nested_dict)

    def check_and_remove_simulations(self, event: Dict) -> Dict:
        """Check to see if any simulations have finished, and remove them
        from the simulation store
        """
        if len(self.simulations) > 0:
            sims_to_delete = []

            for pdb_hash, simulation in self.simulations.items():
                future = simulation["future"]
                pdb_id = simulation["pdb_id"]

                # Check if the future is done
                if future.done():
                    state, error_info = future.result()
                    # If the simulation is done, we can remove it from the simulation store
                    if state == "finished":
                        bt.logging.warning(
                            f"✅ {pdb_id} finished simulation... Removing from execution stack ✅"
                        )
                    else:
                        # If the simulation failed, we should log the error and remove it from the simulation store
                        bt.logging.error(
                            f"❗ {pdb_id} failed simulation... Removing from execution stack ❗"
                        )
                        bt.logging.error(f"Error info: {error_info}")
                    sims_to_delete.append(pdb_hash)

            for pdb_hash in sims_to_delete:
                del self.simulations[pdb_hash]

            running_simulations = [sim["pdb_id"] for sim in self.simulations.values()]

            event["running_simulations"] = running_simulations
            bt.logging.warning(f"Simulations Running: {running_simulations}")

        return event

    def get_simulation_hash(self, pdb_id: str, system_config: Dict) -> str:
        """Creates a simulation hash based on the pdb_id and the system_config given.

        Returns:
            str: first 6 characters of a sha256 hash
        """
        system_hash = pdb_id
        for key, value in system_config.items():
            system_hash += str(key) + str(value)

        hash_object = hashlib.sha256(system_hash.encode("utf-8"))
        return hash_object.hexdigest()[:6]

    def is_unique_job(self, system_config_filepath: str) -> bool:
        """Check to see if a submitted job is unique by checking to see if the folder exists.

        Args:
            system_config_filepath (str): filepath for the config file that specifies the simulation

        Returns:
            bool
        """
        if os.path.exists(system_config_filepath):
            return False
        return True

    def forward(self, synapse: JobSubmissionSynapse) -> JobSubmissionSynapse:
        """
        The main async function that is called by the dendrite to run the simulation.
        There are a set of default behaviours the miner should carry out based on the form the synapse comes in as:

            1. Check to see if the pdb is in the set of simulations that are running
            2. If the synapse md_inputs contains a ckpt file, then we are expected to either accept/reject a simulation rebase. (not implemented yet)
            3. Check if simulation has been run before, and if so, return the files from the last simulation
            4. If none of the above conditions are met, we start a new simulation.
                - If the number of active processes is less than the number of CPUs and the pdb_id is unique, start a new process

        Returns:
            JobSubmissionSynapse: synapse with md_output attached
        """

        pdb_id = synapse.pdb_id

        # If we are already running a process with the same identifier, return intermediate information
        bt.logging.warning(f"⌛ Query from validator for protein: {pdb_id} ⌛")

        # increment step counter everytime miner receives a query.
        self.step += 1
        self.query_start_time = time.time()

        event = self.create_default_dict()
        event["pdb_id"] = pdb_id

        pdb_hash = self.get_simulation_hash(
            pdb_id=pdb_id, system_config=synapse.system_config
        )
        output_dir = os.path.join(self.base_data_path, pdb_id, pdb_hash)
        system_config_filepath = os.path.join(output_dir, f"config_{pdb_id}.pkl")

        # check if any of the simulations have finished
        event = self.check_and_remove_simulations(event=event)
        submitted_job_is_unique = self.is_unique_job(
            system_config_filepath=system_config_filepath
        )

        if submitted_job_is_unique:
            if len(self.simulations) < self.max_workers:
                return self.submit_simulation(
                    synapse=synapse,
                    pdb_id=pdb_id,
                    pdb_hash=pdb_hash,
                    output_dir=output_dir,
                    system_config_filepath=system_config_filepath,
                    event=event,
                )

            elif len(self.simulations) == self.max_workers:
                bt.logging.warning(
                    f"❗ Cannot start new process: job limit reached. ({len(self.simulations)}/{self.max_workers}).❗"
                )

                bt.logging.warning(f"❗ Removing miner from job pool ❗")

                event["condition"] = "cpu_limit_reached"
                synapse.miner_serving = False

                return check_synapse(self=self, synapse=synapse, event=event)

        # The set of RUNNING simulations.
        elif pdb_hash in self.simulations:
            self.simulations[pdb_hash]["queried_at"] = time.time()
            simulation = self.simulations[pdb_hash]
            current_executor_state = simulation["executor"].get_state()
            current_seed = simulation["executor"].seed

            synapse = attach_files_to_synapse(
                synapse=synapse,
                data_directory=simulation["output_dir"],
                state=current_executor_state,
                seed=current_seed,
            )

            event["condition"] = "running_simulation"
            event["state"] = current_executor_state
            event["queried_at"] = simulation["queried_at"]

            return check_synapse(self=self, synapse=synapse, event=event)

        else:
            if os.path.exists(self.base_data_path) and pdb_id in os.listdir(
                self.base_data_path
            ):
                # If we have a pdb_id in the data directory, we can assume that the simulation has been run before
                # and we can return the COMPLETED files from the last simulation. This only works if you have kept the data.

                # We will attempt to read the state of the simulation from the state file
                state_file = os.path.join(output_dir, f"{pdb_id}_state.txt")
                seed_file = os.path.join(output_dir, f"{pdb_id}_seed.txt")

                # Open the state file that should be generated during the simulation.
                try:
                    with open(state_file, "r") as f:
                        lines = f.readlines()
                        state = lines[-1].strip()
                        state = "md_0_1" if state == "finished" else state

                    # If the state is failed, we should not return the files.
                    if state == "failed":
                        synapse.miner_state = state
                        synapse.miner_serving = False
                        event["condition"] = "failed_simulation"
                        event["state"] = state
                        bt.logging.warning(
                            f"❗Returning previous simulation data for failed simulation: {pdb_id}❗"
                        )
                        return check_synapse(self=self, synapse=synapse, event=event)

                    with open(seed_file, "r") as f:
                        seed = f.readlines()[-1].strip()

                    bt.logging.warning(
                        f"❗ Found existing data for protein: {pdb_id}... Sending previously computed, most advanced simulation state ❗"
                    )
                    synapse = attach_files_to_synapse(
                        synapse=synapse,
                        data_directory=output_dir,
                        state=state,
                        seed=seed,
                    )
                except Exception as e:
                    bt.logging.error(
                        f"Failed to read state file for protein {pdb_id} with error: {e}"
                    )
                    state = None

                finally:
                    event["condition"] = "found_existing_data"
                    event["state"] = state

                    return check_synapse(self=self, synapse=synapse, event=event)

            elif len(synapse.md_inputs) == 0:  # The vali sends nothing to the miner
                return check_synapse(self=self, synapse=synapse, event=event)

    def submit_simulation(
        self,
        synapse: JobSubmissionSynapse,
        output_dir: str,
        pdb_id: str,
        pdb_hash: str,
        system_config_filepath: str,
        event: Dict,
    ):
        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=output_dir)

        # We are going to use a hash based on the pdb_id and the system config to index the simulation dict

        # The following files are required for openmm simulations and are received from the validator
        for filename, content in synapse.md_inputs.items():
            write_mode = "w"
            try:
                if "cpt" in filename:
                    write_mode = "wb"
                    content = base64.b64decode(content)

                with open(os.path.join(output_dir, filename), write_mode) as f:
                    f.write(content)

            except Exception as e:
                bt.logging.error(f"Failed to write file {filename!r} with error: {e}")

        # Write the contents of the pdb file to the output directory
        with open(os.path.join(output_dir, f"{pdb_id}.pdb"), "w") as f:
            f.write(synapse.pdb_contents)

        system_config = SimulationConfig(**synapse.system_config)
        write_pkl(system_config, system_config_filepath)

        # Create the job and submit it to the executor
        simulation_manager = SimulationManager(
            pdb_id=pdb_id,
            output_dir=output_dir,
            system_config=system_config.to_dict(),
            seed=self.generate_random_seed()
            if system_config.seed is None
            else system_config.seed,
        )

        future = self.executor.submit(
            simulation_manager.run,
            self.config.mock or self.mock,  # self.mock is inside of MockFoldingMiner
        )

        self.simulations[pdb_hash]["pdb_id"] = pdb_id
        self.simulations[pdb_hash]["executor"] = simulation_manager
        self.simulations[pdb_hash]["future"] = future
        self.simulations[pdb_hash]["output_dir"] = simulation_manager.output_dir
        self.simulations[pdb_hash]["queried_at"] = time.time()

        bt.logging.success(f"✅ New pdb_id {pdb_id} submitted to job executor ✅ ")

        event["condition"] = "new_simulation"
        event["start_time"] = time.time()
        return check_synapse(self=self, synapse=synapse, event=event)

    async def blacklist(self, synapse: JobSubmissionSynapse) -> Tuple[bool, str]:
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            # We also check if the stake is greater than 10_000, which is the minimum stake to not be blacklisted.
            if (
                not self.metagraph.validator_permit[uid]
                or self.metagraph.stake[uid] < 10_000
            ):
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: JobSubmissionSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority


class SimulationManager:
    def __init__(
        self, pdb_id: str, output_dir: str, seed: int, system_config: dict
    ) -> None:
        self.pdb_id = pdb_id
        self.state: str = None
        self.seed = seed
        self.pdb_obj = app.PDBFile(os.path.join(output_dir, f"{pdb_id}.pdb"))

        self.state_file_name = f"{pdb_id}_state.txt"
        self.seed_file_name = f"{pdb_id}_seed.txt"
        self.simulation_steps: dict = system_config["simulation_steps"]
        self.system_config = SimulationConfig(**system_config)

        self.output_dir = output_dir
        self.start_time = time.time()

        self.cpt_file_mapper = {
            "nvt": f"{output_dir}/em.cpt",
            "npt": f"{output_dir}/nvt.cpt",
            "md_0_1": f"{output_dir}/npt.cpt",
        }

        self.STATES = ["nvt", "npt", "md_0_1"]
        self.CHECKPOINT_INTERVAL = 10000
        self.STATE_DATA_REPORTER_INTERVAL = 10
        self.EXIT_REPORTER_INTERVAL = 10

    def create_empty_file(self, file_path: str):
        # For mocking
        with open(file_path, "w") as f:
            pass

    def write_state(self, state: str, state_file_name: str, output_dir: str):
        with open(os.path.join(output_dir, state_file_name), "w") as f:
            f.write(f"{state}\n")

    def run(
        self,
        mock: bool = False,
    ):
        """run method to handle the processing of generic simulations.

        Args:
            simulations (Dict): state_name : OpenMMSimulation object dictionary
            suppress_cmd_output (bool, optional): Defaults to True.
            mock (bool, optional): mock for debugging. Defaults to False.
        """
        bt.logging.info(f"Running simulation for protein: {self.pdb_id}")
        simulations = self.configure_commands(
            seed=self.seed, system_config=copy.deepcopy(self.system_config)
        )
        bt.logging.info(f"Simulations: {simulations}")

        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir)

        # Write the seed so that we always know what was used.
        with open(self.seed_file_name, "w") as f:
            f.write(f"{self.seed}\n")

        try:
            for state, simulation in simulations.items():
                bt.logging.info(f"Running {state} commands")

                self.write_state(
                    state=state,
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir,
                )

                simulation.loadCheckpoint(self.cpt_file_mapper[state])
                simulation.step(self.simulation_steps[state])

                # TODO: Add a Mock pipeline for the new OpenMM simulation here.

            bt.logging.success(f"✅ Finished simulation for protein: {self.pdb_id} ✅")

            state = "finished"
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, None

        # This is the exception that is raised when the simulation fails.
        except mm.OpenMMException as e:
            state = "failed"
            error_info = {
                "type": "OpenMMException",
                "message": str(e),
                "traceback": traceback.format_exc(),  # This is the traceback of the exception
            }
            try:
                platform = mm.Platform.getPlatformByName("CUDA")
                error_info["cuda_version"] = platform.getPropertyDefaultValue(
                    "CudaCompiler"
                )
            except:
                error_info["cuda_version"] = "Unable to get CUDA information"
            finally:
                self.write_state(
                    state=state,
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir,
                )
                return state, error_info

        # Generic Exception
        except Exception as e:
            state = "failed"
            error_info = {
                "type": "UnexpectedException",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, error_info

    def get_state(self) -> str:
        """get_state reads a txt file that contains the current state of the simulation"""
        with open(os.path.join(self.output_dir, self.state_file_name), "r") as f:
            lines = f.readlines()
            return (
                lines[-1].strip() if lines else None
            )  # return the last line of the file

    def get_seed(self) -> str:
        with open(os.path.join(self.output_dir, self.seed_file_name), "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else None

    def configure_commands(
        self, seed: int, system_config: SimulationConfig
    ) -> Dict[str, List[str]]:
        state_commands = {}

        for state in self.STATES:
            simulation, _ = OpenMMSimulation().create_simulation(
                pdb=self.pdb_obj,
                system_config=system_config.get_config(),
                seed=seed,
                state=state,
            )
            simulation.reporters.append(
                LastTwoCheckpointsReporter(
                    file_prefix=f"{self.output_dir}/{state}",
                    reportInterval=self.CHECKPOINT_INTERVAL,
                )
            )
            simulation.reporters.append(
                app.StateDataReporter(
                    file=f"{self.output_dir}/{state}.log",
                    reportInterval=self.STATE_DATA_REPORTER_INTERVAL,
                    step=True,
                    potentialEnergy=True,
                )
            )
            simulation.reporters.append(
                ExitFileReporter(
                    filename=f"{self.output_dir}/{state}",
                    reportInterval=self.EXIT_REPORTER_INTERVAL,
                    file_prefix=state,
                )
            )
            state_commands[state] = simulation

        return state_commands


class MockSimulationManager(SimulationManager):
    def __init__(self, pdb_id: str, output_dir: str) -> None:
        super().__init__(pdb_id=pdb_id)
        self.required_values = set(["init", "wait", "finished"])
        self.output_dir = output_dir

    def run(self, total_wait_time: int = 1):
        start_time = time.time()

        bt.logging.debug(f"✅ MockSimulationManager.run is running ✅")
        check_if_directory_exists(output_directory=self.output_dir)

        store = os.path.join(self.output_dir, self.state_file_name)
        states = ["init", "wait", "finished"]

        intermediate_interval = total_wait_time / len(states)

        for state in states:
            bt.logging.info(f"Running state: {state}")
            state_time = time.time()
            with open(store, "w") as f:
                f.write(f"{state}\n")

            time.sleep(intermediate_interval)
            bt.logging.info(f"Total state_time: {time.time() - state_time}")

        bt.logging.warning(f"Total run method time: {time.time() - start_time}")
