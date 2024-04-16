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

        self.rewards = {}

    def collate_data(self, data: Dict) -> pd.DataFrame:
        self.df = pd.DataFrame()

        for uid, dataset in data.items():
            if dataset is None:  # occurs when status_code is not 200
                continue  # reward is already set to 0.

            subset = dataset[self.name]
            subset["uid"] = uid
            self.df = pd.concat([self.df, subset], axis=0)

        return self.df

    def get_energy(self, df: pd.DataFrame):
        # The dataframe has all data exported by gromacs, and therefore can have different lengths.
        # Different lengths cause NaNs in other cols.

        try:
            subset = df[~df[self.name].isna()]
            min_each_uid = subset.groupby("uid").prod_energy.min()

            best_miner_uid = min_each_uid.idxmin()
            minimum_energy = min_each_uid[best_miner_uid]

            differences = minimum_energy - min_each_uid

            mean_difference = differences.drop(best_miner_uid).mean()
            std_difference = differences.drop(best_miner_uid).std()

            self.rewards[best_miner_uid] = 1

            extra_info = dict(
                minimum_energy=minimum_energy,
                differences=differences.values.tolist(),
                mean_difference=mean_difference,
                std_difference=std_difference,
            )

        except Exception as E:
            bt.logging.error(f"Exception in EnergyRewardModel: {E}")
            extra_info = None

        finally:
            return self.rewards, extra_info

    def reward(self, data: Dict) -> BatchRewardOutput:
        """Apply the necessary steps to get energy reward data

        Args:
            data (Dict): Dict[uid : Dict[str : pd.DataFrame]]
        """

        # need to initialize the reward dictionary with 0s
        for uid in data.keys():
            self.rewards[uid] = 0

        df = self.collate_data(data=data)
        rewards, extra_info = self.get_energy(df=df)

        return BatchRewardOutput(rewards=rewards, extra_info=extra_info)
