from folding.base.reward import BaseReward, BatchRewardOutput, BatchRewardInput
import torch
from loguru import logger
from folding.store import Job
from folding.rewards.linear_reward import divide_decreasing

class FoldingReward(BaseReward):
    """Folding reward class"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def name(self) -> str:
        return 'folding_reward'
        
    
    async def get_rewards(self, data: BatchRewardInput, rewards: torch.Tensor) -> BatchRewardOutput:
        """
        A reward pipeline that determines how to place rewards onto the miners sampled within the batch.
        Currently applies a linearly decreasing reward on all miners that are not the current best / previously
        best loss using the function "divide_decreasing".

        Args:
            energies (torch.Tensor): tensor of returned energies
            rewards (torch.Tensor): tensor of rewards, floats.
            top_reward (float): upper bound reward.
            job (Job)
        """
        energies: torch.Tensor = data.energies
        top_reward: float = data.top_reward
        job: Job = data.job
        
        nonzero_energies: torch.Tensor = torch.nonzero(energies)
        
        info = {
            'name': self.name(),
            'top_reward': top_reward,
        }

        # If the best hotkey is not in the set of hotkeys in the job, this means that the top miner has stopped replying.
        if job.best_hotkey not in job.hotkeys:
            logger.warning(
                f"Best hotkey {job.best_hotkey} not in hotkeys {job.hotkeys}. Assigning no reward."
            )
            return BatchRewardOutput(rewards=rewards, extra_info=info) # rewards of all 0s.

        best_index: int = job.hotkeys.index(job.best_hotkey)

        # There are cases where the top_miner stops replying. ensure to assign reward.
        rewards[best_index] = top_reward

        # If no miners reply, we want *all* reward to go to the top miner.
        if len(nonzero_energies) == 0:
            rewards[best_index] = 1
            return BatchRewardOutput(rewards=rewards, extra_info=info)

        if (len(nonzero_energies) == 1) and (nonzero_energies[0] == best_index):
            rewards[best_index] = 1
            return BatchRewardOutput(rewards=rewards, extra_info=info)

        # Find if there are any indicies that are the same as the best value
        remaining_miners = {}
        for index in nonzero_energies:
            # There could be multiple max energies.
            # The best energy could be the one that is saved in the store.
            if (energies[index] == job.best_loss) or (index == best_index):
                rewards[index] = top_reward
            else:
                remaining_miners[index] = energies[index]

        # The amount of reward that is distributed to the remaining miners MUST be less than the reward given to the top miners.
        num_reminaing_miners = len(remaining_miners)
        if num_reminaing_miners > 1:
            sorted_remaining_miners = dict(
                sorted(remaining_miners.items(), key=lambda item: item[1])
            )  # sort smallest to largest

            # Apply a fixed decrease in reward on the remaining non-zero miners.
            rewards_per_miner = divide_decreasing(
                amount_to_distribute=1 - top_reward,
                number_of_elements=num_reminaing_miners,
            )
            for index, r in zip(sorted_remaining_miners.keys(), rewards_per_miner):
                rewards[index] = r
        else:
            for index in remaining_miners.keys():
                rewards[index] = 1 - top_reward

        return BatchRewardOutput(rewards=rewards, extra_info=info)
    