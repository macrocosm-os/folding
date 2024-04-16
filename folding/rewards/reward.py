import torch
import time
import bittensor as bt
from typing import List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class RewardEvent:
    """Contains rewards for all the responses in a batch"""

    reward_name: str
    rewards: Dict
    batch_time: float
    extra_info: Dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> Dict:
        return {
            f"{self.reward_name}_raw": list(self.rewards.values()),
            f"{self.reward_name}_uids": list(self.rewards.keys()),
            f"{self.reward_name}_batch_time": self.batch_time,
            f"{self.reward_name}_extra_info": self.extra_info,
        }


@dataclass
class BatchRewardOutput:
    rewards: Dict
    extra_info: Dict


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reward(self, data: Dict) -> BatchRewardOutput:
        pass

    @abstractmethod
    def collate_data(self) -> pd.DataFrame:
        pass

    def apply(self, data: Dict) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(data=data)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            reward_name=self.name,
            rewards=batch_rewards_output.rewards,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
