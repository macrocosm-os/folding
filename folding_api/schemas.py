import traceback
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field
from fastapi import Header
import time
from typing import Annotated
from substrateinterface import Keypair
from folding.utils.logger import logger
from hashlib import sha256


class FoldingParams(BaseModel):
    pdb_id: str
    source: Literal["rcsb", "pdbe"]
    ff: str
    water: str
    box: Literal["cube", "box"]
    temperature: float
    friction: float
    epsilon: float


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
    status_codes: List[Any | None] = Field(
        ..., description="The status code of the response."
    )
    job_id: List[str | None] = Field(..., description="The job id of the response.")


class EpistulaHeaders:
    def __init__(
        self,
        version: str = Header(..., alias="Epistula-Version", pattern="^2$"),
        timestamp: str = Header(default=str(time.time()), alias="Epistula-Timestamp"),
        uuid: str = Header(..., alias="Epistula-Uuid"),
        signed_by: str = Header(..., alias="Epistula-Signed-By"),
        request_signature: str = Header(..., alias="Epistula-Request-Signature"),
    ):
        self.version = version
        self.timestamp = timestamp
        self.uuid = uuid
        self.signed_by = signed_by
        self.request_signature = request_signature

    def verify_signature_v2(
        self, body: bytes, now: float
    ) -> Optional[Annotated[str, "Error Message"]]:
        try:
            if not isinstance(self.request_signature, str):
                raise ValueError("Invalid Signature")

            timestamp = int(float(self.timestamp))
            if not isinstance(timestamp, int):
                raise ValueError("Invalid Timestamp")

            if not isinstance(self.signed_by, str):
                raise ValueError("Invalid Sender key")

            if not isinstance(self.uuid, str):
                raise ValueError("Invalid uuid")

            if not isinstance(body, bytes):
                raise ValueError("Body is not of type bytes")

            ALLOWED_DELTA_MS = 8000
            keypair = Keypair(ss58_address=self.signed_by)

            if timestamp + ALLOWED_DELTA_MS < now:
                raise ValueError("Request is too stale")

            message = f"{sha256(body).hexdigest()}.{self.uuid}.{self.timestamp}."
            logger.debug("verifying_signature", message=message)

            verified = keypair.verify(message, self.request_signature)
            if not verified:
                raise ValueError("Signature Mismatch")

            return None

        except Exception as e:
            logger.error(f"signature_verification_failed: {traceback.format_exc()}")
            return str(e)


class APIKeyBase(BaseModel):
    owner: str
    rate_limit: str
    is_active: bool = True


class APIKey(APIKeyBase):
    key: str


class APIKeyCreate(BaseModel):
    owner: str
    rate_limit: str = "100/hour"


class APIKeyResponse(APIKeyBase):
    key: str
