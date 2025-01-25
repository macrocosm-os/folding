from typing import Dict, Any
from folding.base.reward import BaseReward


class SyntheticMDReward(BaseReward):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def calculate_reward(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticMDReward"


class OrganicMDReward(BaseReward):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def calculate_reward(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMDReward"


class SyntheticMLReward(BaseReward):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def calculate_reward(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticMLReward"


class OrganicMLReward(BaseReward):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def calculate_reward(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMLReward"


class RewardRegistry:
    """
    Handles the organization of all tasks that we want inside of SN25, which includes:
        - Molecular Dynamics (MD)
        - ML Inference

    It also attaches its corresponding reward pipelines.
    """

    def __init__(self):
        reward_pipelines = [SyntheticMDReward, OrganicMDReward, SyntheticMLReward, OrganicMLReward]

        self.tasks = {}
        for pipe in reward_pipelines:
            self.tasks[pipe().name()] = pipe
