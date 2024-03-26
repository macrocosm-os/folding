import time
import torch
import argparse
import os
import bittensor as bt

from loguru import logger
from typing import List

from folding.validators.protein import Protein
from folding.utils.uids import get_random_uids
from folding.validators.reward import get_rewards
from folding.protocol import FoldingSynapse


async def run_step(
    self,
    protein: Protein,
    k: int,
    timeout: float,
    task: str = None,
    exclude: list = [],
):
    """
    The function takes a Protein and calculates the molecular dyanmics of it folding without further user input using the GROMACS molecular dynamics software.
    This is done in the following steps:

    1) Stabilizing the protein
    2) Stabilizing the protein in the evironment

    Notes:
    Gromacs is typically run from the command line with heavy user input, the goal of this function is to either skip
    the user input or use different servers to find the optimal input for us based on how "typcial" a run is for each protein

    Gromacs works best for proteins folded in water. Other solutions need a different workflow
    Given that this function makes generic names we may want to have each make a new folder with the proteins name?


    Inputs (main):
    protein_pdb: string arg for filepath of the .pdb file obtained from ___________
    ff: forcefield, this varies by protein and is the second most important input. This is a great use of distributed computed
        and different servers using different force fields is the most optimal way to find this.
    box: constrain the size of the simulated environment to reduce processing speed. Barring vastly irregular shapes,
        the rhombic dodecahedron is most optimal. This could be automated based on known size/shape of the protein or simply relgated to other servers
    energy_min_x: a .mdp file describing the energy minimizatoin parameters. This can be iterated upon based on validation results
        in the tutorial, energy_min_1=emin-charmm.mdp and energy_min_2=nvt-charmm.mdp, energy_min_3=npt-charmm.mdp, energy_min_4=md-charmm.mdp


    Inputs (optional, future):
    solution: Non-water solutions will require different molecular dyanmics functions and workflos
    verbose: Boolean descibing outputs for mdrun. If true, progress updates are ouput while dynamics are calculated
    output: necessary output depends on method of transition state calculations. This can be easily changed

    Final Outputs:
    ________.xvg
    """

    # Second validation checkpoint: After temperature and pressure runs
    # Third validation checkpoint: Post analysis on any metric such as RMSD, radius of gyration, etc.

    bt.logging.debug("run_step")

    # Record event start time.
    event = {"pdb_id": protein.pdb_id, "task": task}

    start_time = time.time()

    # Get the list of uids to query for this step.
    # uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)
    uids = [9, 10]
    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = FoldingSynapse(pdb_id=protein.pdb_id, md_inputs=protein.md_inputs)

    # Make calls to the network with the prompt.
    responses: List[FoldingSynapse] = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )
    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = get_rewards(protein, responses)

    os.system("pm2 stop v1")

    # # Find the best response given the rewards vector.
    # best: str = responses[rewards.argmax(dim=0)]

    # # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # # shape: [ metagraph.n ]
    # scattered_rewards: torch.FloatTensor = self.scores.scatter(0, uids, rewards).to(
    #     self.device
    # )

    # # Update moving_averaged_scores with rewards produced by this step.
    # # shape: [ metagraph.n ]
    # alpha: float = self.config.neuron.moving_average_alpha
    # self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores.to(self.device)

    # # Log the step event.
    # event.update(
    #     {
    #         "block": self.metagraph.block,
    #         "step_length": time.time() - start_time,
    #         "uids": uids.tolist(),
    #         "response_times": [
    #             resp.dendrite.process_time if resp.dendrite.process_time != None else 0
    #             for resp in responses
    #         ],
    #         "response_status_messages": [
    #             str(resp.dendrite.status_message) for resp in responses
    #         ],
    #         "response_status_codes": [
    #             str(resp.dendrite.status_code) for resp in responses
    #         ],
    #         "rewards": rewards.tolist(),
    #         "best": best,
    #     }
    # )

    # bt.logging.debug("event:", str(event))
    # if not self.config.neuron.dont_save_events:
    #     logger.log("EVENTS", "events", **event)


async def forward(self):
    # 1. Select the molecule from online protein database, the forcefield, and the box
    # NOTE: The number of possible inputs should be effectively infinite (or at least very large) so that miners cannot lookup results from earlier runs
    protein = Protein(
        pdb_id=self.config.protein.pdb_id,
        ff=self.config.protein.ff,
        box=self.config.protein.box,
        max_steps=self.config.protein.max_steps,
    )
    bt.logging.info(f"Protein challenge: {protein}")

    # 2. Create the environment and solution the protein is folding in: Preprocess the input files, cleaning up files and generating required inputs
    # 3. Run first step locally
    # First validation checkpoint: After mdrun. Although there are a lot of instances where we can step in for validation, this is the most optimal one for determining run success
    # protein.generate_input_files(run_first_step=self.config.protein.run_first_step)

    # 4. Send the preprocessed inputs and other required details to the miners who will carry out the full MD simulation, score the results and log
    await run_step(
        self,
        protein=protein,
        k=self.config.neuron.sample_size,
        timeout=self.config.neuron.timeout,
    )
