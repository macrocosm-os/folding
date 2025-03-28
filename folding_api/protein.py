from http import HTTPStatus
import json

from fastapi import (
    APIRouter,
    HTTPException,
    Form,
    Request,
    Depends,
    UploadFile,
    File,
    Body,
)
from loguru import logger
from folding_api.schemas import FoldingSchema, FoldingReturn
from folding_api.queries import query_validators
from folding_api.auth import APIKey, get_api_key

router = APIRouter()


async def get_folding_schema(query: str = Form(...)) -> FoldingSchema:
    """
    Dependency function to parse and validate the query form data.
    """
    try:
        query_data = json.loads(query)
        return FoldingSchema(**query_data)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid JSON in query parameter: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid data in query parameter: {str(e)}",
        )


@router.post("/fold")
async def fold(
    request: Request,
    query: FoldingSchema = Body(...),
    api_key: APIKey = Depends(get_api_key),
) -> FoldingReturn:
    """
    Fold with the Bittensor network using a PDB ID from RCSB or PDBE.

    This endpoint is for folding proteins that are already available in the RCSB or PDBE databases.
    The PDB ID must be specified in the query parameters.
    """
    try:
        return await query_validators(query, request.app.state.validator_registry, None)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please have an admin check the logs and try again later.",
        )


@router.post("/fold/upload")
async def fold_with_upload(
    request: Request,
    query: FoldingSchema = Depends(get_folding_schema),
    pdb_file: UploadFile = File(...),
    api_key: APIKey = Depends(get_api_key),
) -> FoldingReturn:
    """
    Fold with the Bittensor network using a custom PDB file.

    This endpoint is for folding proteins using a custom PDB file that you provide.
    The file must be uploaded as part of the request.
    """
    try:
        return await query_validators(
            query, request.app.state.validator_registry, pdb_file
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please have an admin check the logs and try again later.",
        )
