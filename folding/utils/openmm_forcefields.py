from abc import ABC, abstractmethod


class OpenMMForceField(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self):
        ...

    @property
    @abstractmethod
    def recommended_configuration(self):
        ...

    @abstractmethod
    def forcefields(self):
        ...

    @abstractmethod
    def waters(self):
        ...


class Amber14(OpenMMForceField):
    # ref: http://docs.openmm.org/latest/userguide/application/02_running_sims.html?highlight=gaff#amber14

    @property
    def name(self):
        return "amber14"

    @property
    def recommended_configuration(self):
        return ["amber14-all.xml", "amber14/tip3pfb.xml"]

    def forcefields(self):
        forces = [
            "amber14-all.xml",
            "protein.ff14SB.xml",
            "protein.ff15ipq.xml",
            "DNA.OL15.xml",
            "DNA.bsc1.xml",
            "RNA.OL3.xml",
            "lipid17.xml",
            "GLYCAM_06j-1.xml",
        ]
        return [f"{self.name}/{force}" for force in forces]

    def waters(self):
        waters = [
            "tip3p.xml",
            "tip3pfb.xml",
            "tip4pew.xml",
            "tip4pfb.xml",
            "spce.xml",
            "opc.xml",
            "opc3.xml",
        ]
        return [f"{self.name}/{water}" for water in waters]


class Charmm36(OpenMMForceField):
    # ref: http://docs.openmm.org/latest/userguide/application/02_running_sims.html?highlight=gaff#charmm36

    @property
    def name(self):
        return "charmm36"

    @property
    def recommended_configuration(self):
        return ["charmm36.xml", "charmm36/water.xml"]

    def forcefields(self):
        forces = [
            "charmm36.xml",
        ]
        return forces

    def waters(self):
        waters = [
            "water.xml",
            "spce.xml",
            "tip3p-pme-b.xml",
            "tip3p-pme-f.xml",
            "tip4pew.xml",
            "tip4p2005.xml",
            "tip5p.xml",
            "tip5pew.xml",
        ]
        return [f"{self.name}/{water}" for water in waters]


FORCEFIELD_REGISTERY = {"amber14": Amber14, "charmm36": Charmm36}
