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
    rewards: List
    timings: List
    batch_time: float
    extra_info: dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.reward_name}_raw": self.rewards.tolist(),
            f"{self.reward_name}_timings": self.timings.tolist(),
            f"{self.reward_name}_batch_time": self.batch_time,
            f"{self.reward_name}_extra_info": self.extra_info,
        }


@dataclass
class BatchRewardOutput:
    rewards: List
    timings: List
    extra_info: dict

    def __post_init__(self):
        if self.rewards.shape != self.timings.shape:
            raise ValueError(
                f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}"
            )

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (
            self.rewards.max() - self.rewards.min() + 1e-6
        )


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

    def apply(self, df: pd.DataFrame, reward_type: str) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(df)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            reward_name=self.name,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
