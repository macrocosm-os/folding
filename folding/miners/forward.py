import os
import glob
import base64
from threading import Thread, Event
from typing import Dict
import bittensor as bt
import time
import subprocess 

from folding.utils.ops import run_cmd_commands, check_if_directory_exists
from utils.monitor import auto_plot, update_control_file
from folding.protocol import FoldingSynapse

# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def forward(synapse: FoldingSynapse, config: Dict) -> FoldingSynapse:
    # TODO: Determine how many steps to run based on timeout
    bt.logging.info(
        f"Running GROMACS simulation for protein: {synapse.pdb_id} with files {synapse.md_inputs.keys()} mdrun_args: {synapse.mdrun_args}"
    )

    synapse.md_output = {}

    output_directory = os.path.join(
        ROOT_DIR,
        "data",
        "miners",
        synapse.axon.hotkey[:8],
        synapse.pdb_id,
    )

    # Make sure the output directory exists and if not, create it
    check_if_directory_exists(output_directory=output_directory)
    os.chdir(output_directory)

    subprocess.run([F"{ROOT_DIR}/utils/update.sh", output_directory])


    # The following files are required for GROMACS simulations and are recieved from the validator
    for filename, content in synapse.md_inputs.items():
        # Write the file to the output directory
        with open(filename, "w") as file:
            bt.logging.info(f"\nWriting {filename} to {output_directory}")
            file.write(content)

    stop_event = Event() # Create an event to signal the plotting thread to stop

    plotting_thread = Thread(target=auto_plot, args=(output_directory, stop_event))
    plotting_thread.start()

    try:
            bt.logging.info("Starting NVT stage simulation")
            update_control_file('nvt', output_directory)
            nvt_commands = [
                "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",  # Temperature equilibration
                "gmx mdrun -deffnm nvt " + synapse.mdrun_args,
            ]
            run_cmd_commands(nvt_commands, suppress_cmd_output=config.neuron.suppress_cmd_output)

            bt.logging.info("Starting NPT stage simulation")
            update_control_file('npt', output_directory)
            npt_commands = [
                "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",  # Pressure equilibration
                "gmx mdrun -deffnm npt " + synapse.mdrun_args,
            ]
            run_cmd_commands(npt_commands, suppress_cmd_output=config.neuron.suppress_cmd_output)

            bt.logging.info("Starting MD stage simulation")
            update_control_file('md_0_1', output_directory)
            md_commands = [
                "gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr",  # Production run
                "gmx mdrun -deffnm md_0_1 " + synapse.mdrun_args,
                "echo '1\n1\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_center.xtc -center -pbc mol",
            ]
            run_cmd_commands(md_commands, suppress_cmd_output=config.neuron.suppress_cmd_output)

    except Exception as e:
        bt.logging.error(f"Error during simulation: {e}")
        raise
    
    finally:
        # Signal the plotting thread to stop
        stop_event.set()
        plotting_thread.join()  # Wait for the plotting thread to finish

        # Set stop signal for Bash script
        stop_signal_path = f"{output_directory}/stop_signal.txt"
        with open(stop_signal_path, 'w') as file:
            file.write("stop")
        
        # Wait for the Bash script to detect the stop signal and clean up
        time.sleep(10)  

        # Clean up the stop signal file
        if os.path.exists(stop_signal_path):
            os.remove(stop_signal_path)

        bt.logging.info("Simulation and all monitoring have been cleanly stopped.")


    bt.logging.success("Simulation completed successfully")

    # load the output files as bytes and add to synapse.md_output
    desired_files = glob.glob("md_0_1*") + glob.glob("*.edr")
    bt.logging.warning(f"Desired files to send to the validator: {desired_files}")
    for filename in desired_files:
        bt.logging.info(f"Attaching file: {filename!r} to synapse.md_output")
        try:
            with open(filename, "rb") as f:
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            bt.logging.error(f"Failed to read file {filename!r} with error: {e}")

    bt.logging.success(
        f"Finished running GROMACS simulation for protein: {synapse.pdb_id} and attached {len(synapse.md_output)} files to synapse.md_output"
    )
