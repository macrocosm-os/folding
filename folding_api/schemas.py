from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field


class FoldingParams(BaseModel):
    pdb_id: str
    source: Literal["rcsb", "pdbe"]
    ff: str
    water: str
    box: Literal["cube", "box"]
    temperature: float
    friction: float


class FoldingSchema(BaseModel):
    """
    Represents a request to a validator.
    """

    pdb_id: str = Field(
        "1ubq", description="The PDB identifier for the selected response source."
    )
    source: Literal["rcsb", "pdbe"] = Field(
        "rcsb",
        description="The source of the folding request from two recognized databases.",
    )
    ff: str = Field(
        "charmm36.xml", description="The force field for the selected response source."
    )
    water: str = Field(
        "charmm36/water.xml",
        description="The water model for the selected response source.",
    )
    box: Literal["cube", "box"] = Field(
        ..., description="The box type for the selected response source."
    )
    temperature: float = Field(
        ..., description="The temperature for the selected response source."
    )
    friction: float = Field(
        ..., description="The friction coefficient for the selected response source."
    )
    epsilon: float = Field(
        ...,
        description="The base epsilon that should be used for the challenge. Represented in %/100",
    )

    validator_uids: list[int] = Field(
        ..., description="The validator identifier for the selected response source."
    )
    num_validators_to_sample: Optional[int] = Field(
        None, description="The number of validators to sample."
    )

    timeout: int = Field(5, description="The time in seconds to wait for a response.")

    @property
    def folding_params(self) -> FoldingParams:
        """Get the simulation parameters for SimulationConfig"""
        return FoldingParams(
            pdb_id=self.pdb_id,
            source=self.source,
            ff=self.ff,
            water=self.water,
            box=self.box,
            temperature=self.temperature,
            friction=self.friction,
            epsilon=self.epsilon,
        )

    @property
    def api_parameters(self):
        """Get the API parameters for the Folding API"""
        return {
            "validator_uids": self.validator_uids,
            "num_validators_to_sample": self.num_validators_to_sample,
        }


class FoldingReturn(BaseModel):
    """
    Represents a response from a validator.
    """

    uids: List[int] = Field(..., description="The uids of the response.")
    hotkeys: List[str] = Field(..., description="The hotkeys of the response.")
    status_codes: List[Any] = Field(..., description="The status code of the response.")
    job_id: Optional[str] = Field(None, description="The job id of the response.")
