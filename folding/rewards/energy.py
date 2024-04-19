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

    def get_energy(self, df: pd.DataFrame):
        # The dataframe has all data exported by gromacs, and therefore can have different lengths.
        # Different lengths cause NaNs in other cols.

        curves = []
        steps = []
        try:
            subset = df[~df[self.name].isna()]

            for uid in subset.uid.unique():
                s = subset[subset.uid == uid]
                curves.append(list(s[self.name].values))
                steps.append(list(s["step"].values))

            min_each_uid = subset.groupby("uid")[self.name].min()

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
                curves=curves,
                steps=steps,
            )

        except Exception as E:
            bt.logging.error(f"Exception in EnergyRewardModel: {E}")
            extra_info = None

        finally:
            return self.rewards, extra_info

    def get_rewards(self, data: Dict) -> BatchRewardOutput:
        """Apply the necessary steps to get energy reward data

        Args:
            data (Dict): Dict[uid : Dict[str : pd.DataFrame]]
        """

        df = self.collate_data(data=data)
        rewards, extra_info = self.get_energy(df=df)

        return BatchRewardOutput(rewards=rewards, extra_info=extra_info)
