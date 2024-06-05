import torch
from folding.store import Job
from folding.rewards.linear_reward import divide_decreasing


def reward_pipeline(
    energies: torch.Tensor, rewards: torch.Tensor, top_reward: float, job: Job
):
    """A reward pipeline that determines how to place rewards onto the miners sampled within the batch.
    Currently applies a linearly decreasing reward on all miners that are not the current best / previously
    best loss using the function "divide_decreasing".

    Args:
        energies (torch.Tensor): tensor of returned energies
        rewards (torch.Tensor): tensor of rewards, floats.
        top_reward (float): upper bound reward.
        job (Job)
    """
    nonzero_energies = torch.nonzero(energies)
    best_index = job.hotkeys.index(job.best_hotkey)

    # There are cases where the top_miner stops replying. ensure to assign reward.
    rewards[best_index] = top_reward

    # If no miners reply, we want *all* reward to go to the top miner.
    if len(nonzero_energies) == 0:
        rewards[best_index] = 1
        return rewards

    if (len(nonzero_energies) == 1) and (nonzero_energies[0] == best_index):
        rewards[best_index] = 1
        return rewards

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

    return rewards
