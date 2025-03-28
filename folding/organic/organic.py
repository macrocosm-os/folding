from typing import Optional
import uuid
import time
import json
import tempfile
from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File, Form
from folding_api.schemas import EpistulaHeaders
from folding_api.schemas import FoldingParams
from folding.utils.logging import logger


router = APIRouter()


def verify_organic_request(
    request: Request,
    job: FoldingParams,
    epistula_headers: EpistulaHeaders,
) -> None:
    """
    Verify the organic request signature and whitelist.
    """
    body_bytes = json.dumps(job.model_dump(), default=str, sort_keys=True).encode(
        "utf-8"
    )
    try:
        error = epistula_headers.verify_signature_v2(body_bytes, float(time.time()))
        if error:
            raise HTTPException(status_code=403, detail=error)
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

    sender_hotkey = epistula_headers.signed_by
    if sender_hotkey not in request.app.state.config.organic_whitelist:
        logger.warning(
            f"Received organic request from {sender_hotkey}, but {sender_hotkey} is not in the whitelist."
        )
        raise HTTPException(
            status_code=403, detail="Forbidden, sender not in whitelist."
        )


def get_folding_params(query: str = Form(...)) -> FoldingParams:
    """
    Dependency function to parse and validate the query form data.
    """
    try:
        query_data = json.loads(query)
        return FoldingParams(**query_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid query data: {str(e)}")


@router.post("/organic")
async def organic(
    request: Request,
    job: FoldingParams,
    epistula_headers: EpistulaHeaders = Depends(EpistulaHeaders),
):
    """
    This endpoint is used to receive organic requests for proteins from RCSB or PDBE databases.
    Returns success message with the job id.

    Args:
        request: Request
        job: FoldingParams
        epistula_headers: EpistulaHeaders
    Returns:
        dict[str, str]: dict with the job id.
    """
    verify_organic_request(request, job, epistula_headers)

    folding_params = job.model_dump()
    folding_params["job_id"] = str(uuid.uuid4())

    logger.info(f"Received organic request: {folding_params}")
    request.app.state.validator._organic_queue.add(folding_params)

    return {"job_id": folding_params["job_id"]}


@router.post("/organic/upload")
async def organic_with_upload(
    request: Request,
    job: FoldingParams = Depends(get_folding_params),
    pdb_file: UploadFile = File(...),
    epistula_headers: EpistulaHeaders = Depends(EpistulaHeaders),
):
    """
    This endpoint is used to receive organic requests with custom PDB files.
    Returns success message with the job id.

    Args:
        request: Request
        job: FoldingParams
        pdb_file: PDB file upload
        epistula_headers: EpistulaHeaders
    Returns:
        dict[str, str]: dict with the job id.
    """
    verify_organic_request(request, job, epistula_headers)

    folding_params = job.model_dump()
    folding_params["job_id"] = str(uuid.uuid4())

    # Handle PDB file
    try:
        # Create a temporary file to store the PDB
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdb", delete=False
        ) as temp_pdb:
            content = await pdb_file.read()
            temp_pdb.write(content)
            temp_pdb_path = temp_pdb.name

        # Update folding params with the temporary file path
        folding_params["pdb_file_path"] = temp_pdb_path
        logger.info(f"Created temporary PDB file at: {temp_pdb_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDB file: {str(e)}")

    logger.info(f"Received organic request with PDB file: {folding_params}")
    request.app.state.validator._organic_queue.add(folding_params)

    return {"job_id": folding_params["job_id"]}
