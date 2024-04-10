import time
import torch
import bittensor as bt
from typing import List, Tuple, Dict

from folding.validators.protein import Protein
from folding.utils.uids import get_random_uids
from folding.utils.logging import log_event
from folding.validators.reward import get_rewards
from folding.protocol import FoldingSynapse

from folding.validators.hyperparameters import HyperParameters


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

    Inputs (main):
    protein_pdb: string arg for name of the .pdb file
    ff: forcefield, this varies by protein and is the second most important input. This is a great use of distributed computed
        and different servers using different force fields is the most optimal way to find this.
    box: constrain the size of the simulated environment to reduce processing speed. Barring vastly irregular shapes,
        the rhombic dodecahedron is most optimal. This could be automated based on known size/shape of the protein or simply relgated to other servers
    energy_min_x: an .mdp file describing the energy minimization parameters. This can be iterated upon based on validation results
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
    uids = [9]
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

    # # Log the step event.
    event.update(
        {
            "block": self.metagraph.block,
            "step_length": time.time() - start_time,
            "uids": uids.tolist(),
            "response_times": [
                resp.dendrite.process_time if resp.dendrite.process_time != None else 0
                for resp in responses
            ],
            "response_status_messages": [
                str(resp.dendrite.status_message) for resp in responses
            ],
            "response_status_codes": [
                str(resp.dendrite.status_code) for resp in responses
            ],
            # "rewards": rewards.tolist(),
        }
    )

    return event

    # os.system("pm2 stop v1")

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

    # bt.logging.debug("event:", str(event))
    # if not self.config.neuron.dont_save_events:
    #     logger.log("EVENTS", "events", **event)


def parse_config(config) -> List[str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """
    ff = config.protein.ff
    water = config.protein.water
    box = config.protein.box
    exclude_in_hp_search = []

    if ff is not None:
        exclude_in_hp_search.append("ff")
    if water is not None:
        exclude_in_hp_search.append("water")
    if box is not None:
        exclude_in_hp_search.append("box")

    return exclude_in_hp_search


async def forward(self):
    forward_start_time = time.time()
    exclude_in_hp_search = parse_config(self.config)

    hp_sampler = HyperParameters(
        pdb_id=self.config.protein.pdb_id,
        exclude=exclude_in_hp_search,
    )

    try:
        sampled_combination: Dict[str, Dict] = hp_sampler.sample_hyperparameters()
        hp: Dict = sampled_combination[self.config.protein.pdb_id]

        protein = Protein(
            pdb_id=self.config.protein.pdb_id,
            ff=self.config.protein.ff
            if self.config.protein.ff is not None
            else hp["ff"],
            water=self.config.protein.water
            if self.config.protein.water is not None
            else hp["water"],
            box=self.config.protein.box
            if self.config.protein.box is not None
            else hp["box"],
            config=self.config.protein,
        )

        bt.logging.info(f"Attempting to generate challenge: {protein}")
        protein.forward()

    except Exception as E:
        bt.logging.error(f"Error running hyperparameters {hp}")

    event = await run_step(
        self,
        protein=protein,
        k=self.config.neuron.sample_size,
        timeout=self.config.neuron.timeout,
    )

    event["forward_time"] = time.time() - forward_start_time
    event["pdb_id"] = self.config.protein.pdb_id
    event.update(hp)  # add the protein hyperparameters to the event

    log_event(self, event)
