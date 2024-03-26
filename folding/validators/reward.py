import torch
import bittensor as bt
from typing import List

from .protein import Protein
from folding.protocol import FoldingSynapse


def get_rewards(protein: Protein, responses: List[FoldingSynapse]) -> torch.FloatTensor:
    """Applies the reward model across each call. Unsuccessful responses are zeroed."""
    # Get indices of correctly responding calls.

    for idx, resp in enumerate(responses):
        md_output_summary = {k: len(v) for k, v in resp.md_output.items()}
        bt.logging.info(
            f"Response {idx}:\nDendrite: {resp.dendrite}\nMD Output: {md_output_summary}\n"
        )

    successful_response_indices: List[int] = [
        idx for idx, resp in enumerate(responses) if resp.dendrite.status_code == 200
    ]
    bt.logging.success(f"successful_response_indices: {successful_response_indices}")

    # Get all responses from responding calls.
    successful_responses: List[str] = [
        responses[idx] for idx in successful_response_indices
    ]

    # Reward each response.
    successful_rewards = [
        protein.reward(md_output=resp.md_output, hotkey=resp.dendrite.hotkey)
        for resp in successful_responses
    ]

    bt.logging.success(f"successful_rewards: {successful_rewards}")

    # Softmax rewards across samples.
    successful_rewards_normalized = torch.softmax(
        torch.tensor(successful_rewards), dim=0
    )

    # Init zero rewards for all calls.
    filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
    filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

    # Fill reward tensor.
    for idx, reward, reward_normalized in zip(
        successful_response_indices,
        successful_rewards,
        successful_rewards_normalized,
    ):
        filled_rewards[idx] = reward
        filled_rewards_normalized[idx] = reward_normalized

    # Return the filled rewards.
    return filled_rewards, filled_rewards_normalized
