import difflib
import torch

import pandas as pd
from folding.rewards.reward import BaseRewardModel, BatchRewardOutput
from typing import List, Dict


class EnergyRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "prod_energy"

    def __init__(self, data: Dict[int : Dict[str : pd.DataFrame]], **kwargs):
        super().__init__()
        self.data = data

    def collate_data(self) -> pd.DataFrame:
        self.df = pd.DataFrame()

        for uid, dataset in self.data.items():
            subset = pd.concat(dataset[self.name], ignore_index=True, axis=0)
            subset["uid"] = uid
            self.df = pd.concat([self.df, subset], axis=0)

        return self.df

    def get_energy(self, df: pd.DataFrame):
        subset = df[~df[self.name].isna()]
        minimum = subset.groupby("uid").prod_energy.min()

        best_miner_uid = minimum[minimum == minimum.min()].index[0]
        minimum_energy = subset[best_miner_uid]
        differences = minimum_energy - subset

        mean_difference = differences.drop(best_miner_uid).mean()
        std_difference = differences.drop(best_miner_uid).std()

        return (
            best_miner_uid,
            minimum_energy,
            differences,
            mean_difference,
            std_difference,
        )

    def reward(self):
        df = self.collate_data()
        (
            best_miner_uid,
            minimum_energy,
            differences,
            mean_difference,
            std_difference,
        ) = self.get_energy(df=df)

        BatchRewardOutput(rewards=rewards, timings=timings, extra_info={"Testing": 10})
