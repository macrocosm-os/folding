import pandas as pd
from typing import List, Dict
import bittensor as bt

from folding.rewards.reward import BaseRewardModel, BatchRewardOutput


class EnergyRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "prod_energy"

    def __init__(self, **kwargs):
        super().__init__()

    def collate_data(self, data: Dict) -> pd.DataFrame:
        self.df = pd.DataFrame()

        for uid, dataset in data.items():
            subset = dataset[self.name]
            subset["uid"] = uid
            self.df = pd.concat([self.df, subset], axis=0)

        return self.df

    def get_energy(self, df: pd.DataFrame):
        bt.logging.info(f"df: {df}")
        subset = df[~df[self.name].isna()]

        uids = subset.uid.unique().tolist()
        min_each_uid = subset.groupby("uid").prod_energy.min()

        bt.logging.info(f"Min uid each uid: {min_each_uid}")

        best_miner_uid = min_each_uid.idxmin()
        minimum_energy = min_each_uid[best_miner_uid]

        differences = minimum_energy - min_each_uid

        mean_difference = differences.drop(best_miner_uid).mean()
        std_difference = differences.drop(best_miner_uid).std()

        # Simple reward schema where all miners get 0 except the winner.
        rewards = {uid: 0 for uid in uids}
        rewards[best_miner_uid] = 1

        return (
            rewards,
            minimum_energy,
            differences,
            mean_difference,
            std_difference,
        )

    def reward(self, data: Dict) -> BatchRewardOutput:
        """Apply the necessary steps to get energy reward data

        Args:
            data (Dict): Dict[int : Dict[str : pd.DataFrame]]
        """
        df = self.collate_data(data=data)
        (
            rewards,
            minimum_energy,
            differences,
            mean_difference,
            std_difference,
        ) = self.get_energy(df=df)

        extra_info = {
            "minimum_energy": minimum_energy,
            "differences": differences,
            "mean_difference": mean_difference,
            "std_difference": std_difference,
        }

        return BatchRewardOutput(rewards=rewards, extra_info=extra_info)
