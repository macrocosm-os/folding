import torch
import time
import bittensor as bt
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class RewardEvent:
    """Contains rewards for all the responses in a batch"""

    reward_name: str
    rewards: Dict
    batch_time: float

    extra_info: Optional[Dict] = None

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> Dict:
        d = {
            f"{self.reward_name}_raw": list(self.rewards.values()),
            f"{self.reward_name}_uids": list(self.rewards.keys()),
            f"{self.reward_name}_batch_time": self.batch_time,
        }

        if self.extra_info is not None:
            d[f"{self.reward_name}_extra_info"] = self.extra_info

        return d


@dataclass
class BatchRewardOutput:
    rewards: Dict
    extra_info: Optional[Dict] = None


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_rewards(self, data: Dict) -> BatchRewardOutput:
        pass

    def setup_rewards(self, data: Dict):
        """Sets a default dict for all RewardModels"""
        rewards = {}
        for uid in data.keys():
            rewards[uid] = 0

        return rewards

    def collate_data(self, data: Dict) -> pd.DataFrame:
        """collect the desired data for a chosen reward model.

        Args:
            data (Dict): Dictionary mapping between uid : Dict[self.name : pd.DataFrame]

        Returns:
            pd.DataFrame: Collected data across all uids for the self.name property
        """
        self.df = pd.DataFrame()

        for uid, dataset in data.items():
            if dataset is None:  # occurs when status_code is not 200
                continue  # reward is already set to 0.

            subset = dataset[self.name]
            subset["uid"] = uid
            self.df = pd.concat([self.df, subset], axis=0)

        return self.df  # if no miners return data, then this is an empty dataframe.

    def apply(self, data: Dict) -> RewardEvent:
        self.rewards = self.setup_rewards(data=data)

        t0 = time.time()
        batch_rewards_output = self.get_rewards(data=data)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            reward_name=self.name,
            rewards=batch_rewards_output.rewards,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
