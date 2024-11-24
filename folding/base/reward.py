from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Optional
import time
import torch
from folding.store import Job


class RewardEvent(BaseModel):
    """Contains rewards for all the responses in a batch"""

    reward_name: str
    rewards: torch.Tensor
    batch_time: float

    extra_info: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class BatchRewardOutput(BaseModel):
    rewards: torch.Tensor
    extra_info: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class BatchRewardInput(BaseModel):
    energies: torch.Tensor
    top_reward: float
    job: Job

    class Config:
        arbitrary_types_allowed = True


class BaseReward(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    async def get_rewards(
        self, data: BatchRewardInput, rewards: torch.Tensor
    ) -> BatchRewardOutput:
        pass

    async def setup_rewards(self, energies: torch.Tensor) -> torch.Tensor:
        """Setup rewards for the given energies"""
        return torch.zeros(len(energies))

    async def apply(self, data: BatchRewardInput) -> RewardEvent:
        self.rewards: torch.Tensor = await self.setup_rewards(energies=data.energies)
        t0: float = time.time()
        batch_rewards_output: BatchRewardOutput = await self.get_rewards(
            data=data, rewards=self.rewards
        )
        batch_rewards_output.rewards = await self.calculate_final_reward(
            rewards=batch_rewards_output.rewards, job=data.job
        )
        batch_rewards_time: float = time.time() - t0

        return RewardEvent(
            reward_name=self.name(),
            rewards=batch_rewards_output.rewards,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
        )

    async def calculate_final_reward(
        self, rewards: torch.Tensor, job: Job
    ) -> torch.Tensor:
        # priority_multiplier = 1 + (job.priority - 1) * 0.1 TODO: Implement priority
        priority_multiplier = 1.0
        organic_multiplier = 1.0
        if "is_organic" in job.event.keys():
            if job.event["is_organic"]:
                organic_multiplier = 10.0

        return rewards * priority_multiplier * organic_multiplier

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
