import subprocess
import tqdm
from typing import List, Dict
from collections import defaultdict
import bittensor as bt 

"""
We need something thet gets returned (result dict) 
We don't want it to grow each time its called. (self) 
We want it to be independent of gromacs
We dont need to be worried about pdb_id 
This needs to log to the event dict. 
"""


def output_dict():
    return defaultdict(str)

class RunAndLog():
    def __init__(self):
        self.command_dict = defaultdict(output_dict)
    
    def run_cmd_commands(
            self, commands: List [str], suppress_cmd_output: bool = True, verbose: bool = False
    ) -> Dict[str, str]:
        for cmd in tqdm.tqdm(commands): # set tuple 
            bt.logging.debug(F"Running command {cmd}")

            try:
                # result = run the command (cmd), check that it succedded, executed it through the shell, captures its output and error messages. 
                result = subprocess.run( 
                    cmd, 
                    check=True, # Ensuring that the command is run successfully
                    shell=True, # cmd will be executed through the shell
                    stdout=subprocess.PIPE, # Attributes for the output of the command
                    stderr=subprocess.PIPE, # Attributes for the error of the command
                )
                if not suppress_cmd_output:
                    bt.logging.info(result.stdout.decode())

                # UPDATE DICT HERE 
                self.command_dict[str(cmd)]["status"] = 'success'
                self.command_dict[str(cmd)]["success_output"] = result.stdout.decode()

            except subprocess.CalledProcessError as e:
                bt.logging.error(f"❌ Failed to run command ❌: {cmd}")
                if verbose:
                    bt.logging.error(f"Output: {e.stdout.decode()}")
                    bt.logging.error(f"Error: {e.stderr.decode()}")

                # UPDATE DICT HERE (probably wise to use the stdout and stderr attributes of the result object)
                self.command_dict[str(cmd)]["status"] = 'failed'
                self.command_dict[str(cmd)]["error"] = e.stderr.decode()
                self.command_dict[str(cmd)]["failed_output"] = e.stdout.decode()


current_directory = '/home/spunion/folding/data/8emf/deliverable1/no_seed_1'
gmx_command_1 = 'gmx grompp -f e.mdp -c 8.gro -p topol.top -o e.tpr -maxwarn 100'
gmx_command_2 = 'gmx mdrun -v -deffnm emin '

test_commands = ['cd /home/spunion/folding/data/8emf/deliverable1/no_seed_1', gmx_command_1, gmx_command_2]

RunAndLog_instance = RunAndLog()
RunAndLog_instance.run_cmd_commands(commands = test_commands, suppress_cmd_output = True, verbose = False)

print(RunAndLog_instance.command_dict)