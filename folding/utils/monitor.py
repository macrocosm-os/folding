import pandas as pd
import plotly.express as px
import time
import os
import subprocess
import re
import multiprocessing
from folding.utils.ops import run_cmd_commands

# identify the root directory 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def get_running_pdbs():
    pass

def prepare_plot_files(pdb_directory: str):
    # use the directory find the edr files and make them into .xvg files
    for files in pdb_directory: 
        # if the file has extension edr
        if files.endswith(".edr"):

            edr_dict = {} # create an empty dict to which we will append the edr files with their time stamps
            for file in files:
                current_time = time.getmtime(files) # get the time of the most recent edr file
                edr_dict[file]= current_time # add the edr files with their time stamps
                most_recent_edr = max(edr_dict) # find the most recent edr file
                most_recent_edr_str = most_recent_edr[0].split(".")[0] # strip the extension

            commands = [
                f"gmx energy -f {most_recent_edr} -o '{most_recent_edr_str}_potential_.xvg' -nobackup",
            ]
            
            run_cmd_commands(commands,suppress_cmd_output=True)

            time.sleep(1) # sleep for 1 second
        return most_recent_edr, most_recent_edr_str

def auto_plot(pdb_directory: str, output_file: str, most_recent_edr_str: str):
    while True: 
        try:
            file_path = os.path.join(pdb_directory, f"{most_recent_edr_str}_potential.xvg")
            df = pd.read_csv(file_path, skiprows=24, sep="\s+", names=["time", "potential_energy"])
            custom_labels = {
            "time": "Step (0.01ps/step)",
            "potential_energy": "Potential Energy (kJ/mol)"
            }
            fig = px.line(df, x="time", y="potential_energy", title="8emf.pdb - Potential Energy vs Time (Step)",template='plotly_white', labels=custom_labels)
            fig.write_image(os.path.join(pdb_directory, output_file))
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)


def Monitor():
    data_location = f"{ROOT_DIR}/data/{pdb_directory}"
    output_file = f"potential_energy_vs_time_{most_recent_edr_str}.png"

    
    prepare_process = multiprocessing.Process(target=prepare_plot_files, args=(data_location,))# Create a Process for prepare_plot_files
    
    prepare_process.start()# Start the prepare_plot_files process

    prepare_process.join()# Wait for prepare_plot_files to finish and get its return values
    most_recent_edr, most_recent_edr_str = prepare_process.exitcode

    auto_plot_process = multiprocessing.Process(target=auto_plot, args=(data_location, output_file, most_recent_edr_str))    # Create a Process for auto_plot
    
    auto_plot_process.start()# Start the auto_plot process

    auto_plot_process.join()    # Optionally, wait for auto_plot to finish