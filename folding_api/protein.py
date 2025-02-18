from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Body
from loguru import logger
from folding_api.schemas import FoldingSchema, FoldingReturn

router = APIRouter()


@router.post("/fold")
async def fold(query: FoldingSchema = Body(...)) -> FoldingReturn:
    """
    fold with the Bittensor network.

    This request is meant to be a *submission* of a folding job to the network.
    As such, you are expected to get a quick 200-level response back.
    """

    try:
        server_status.check_subnet_status(Subnet.SN25.value)
        return await query_validators(query)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please have an admin check the logs and try again later.",
        )
