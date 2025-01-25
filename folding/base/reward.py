from abc import ABC, abstractmethod


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
