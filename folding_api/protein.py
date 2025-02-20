from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Body, Request
from loguru import logger
from folding_api.schemas import FoldingSchema, FoldingReturn
from folding_api.queries import query_validators

router = APIRouter()


@router.post("/fold")
async def fold(request: Request, query: FoldingSchema = Body(...)) -> FoldingReturn:
    """
    fold with the Bittensor network.

    This request is meant to be a *submission* of a folding job to the network.
    As such, you are expected to get a quick 200-level response back.
    """

    try:
        return await query_validators(query, request.app.state.validator_registry)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please have an admin check the logs and try again later.",
        )
