from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEvaluator(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def _evaluate(self) -> float:
        pass

    @abstractmethod
    def _validate(self):
        pass

    def forward(self, data: Dict[str, Any]) -> Any:
        return self._evaluate(data=data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
