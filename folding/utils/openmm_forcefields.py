from abc import ABC, abstractmethod
from typing import Dict, List


class OpenMMForceField(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def recommended_configuration(self) -> Dict[str, str]:
        ...

    @property
    @abstractmethod
    def forcefields(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def waters(self) -> List[str]:
        ...

    def _prefix_paths(self, items: List[str]) -> List[str]:
        return [f"{self.name}/{item}" for item in items]


class Amber14(OpenMMForceField):
    name = "amber14"
    recommended_configuration = {
        "FF": "amber14-all.xml",
        "WATER": "amber14/tip3pfb.xml",
        "BOX": "cube",
    }

    @property
    def forcefields(self):
        return self._prefix_paths(
            [
                "protein.ff14SB.xml",
                "protein.ff15ipq.xml",
                "DNA.OL15.xml",
                "DNA.bsc1.xml",
                "RNA.OL3.xml",
                "lipid17.xml",
            ]
        )

    @property
    def waters(self):
        return self._prefix_paths(
            [
                "tip3p.xml",
                "tip3pfb.xml",
                "tip4pew.xml",
                "tip4pfb.xml",
                "spce.xml",
                "opc.xml",
                "opc3.xml",
            ]
        )


class Charmm36(OpenMMForceField):
    name = "charmm36"
    recommended_configuration = {
        "FF": "charmm36.xml",
        "WATER": "charmm36/water.xml",
        "BOX": "cube",
    }

    @property
    def forcefields(self):
        return ["charmm36.xml"]

    @property
    def waters(self):
        return self._prefix_paths(
            [
                "water.xml",
                "spce.xml",
                "tip3p-pme-b.xml",
                "tip3p-pme-f.xml",
                "tip4pew.xml",
                "tip4p2005.xml",
                "tip5p.xml",
                "tip5pew.xml",
            ]
        )


FORCEFIELD_REGISTRY = {cls.name: cls for cls in [Amber14, Charmm36]}
