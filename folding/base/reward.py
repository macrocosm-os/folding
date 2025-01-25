from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseReward(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        pass

    def forward(self, data: Dict[str, Any]) -> float:
        return self.calculate_reward(data=data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
