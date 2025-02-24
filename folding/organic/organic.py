import uuid
import time
import json
from fastapi import APIRouter, Request, Depends, HTTPException
from folding_api.schemas import EpistulaHeaders
from folding_api.schemas import FoldingParams
from folding.utils.logging import logger


router = APIRouter()


@router.post("/organic")
async def organic(
    request: Request,
    job: FoldingParams,
    epistula_headers: EpistulaHeaders = Depends(EpistulaHeaders),
):
    """
    This endpoint is used to receive organic requests. Returns success message with the job id.
    Args:
        request: Request
        job: FoldingParams
        epistula_headers: EpistulaHeaders
    Returns:
        dict[str, str]: dict with the job id.
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

    folding_params = job.model_dump()
    folding_params["job_id"] = str(uuid.uuid4())
    logger.info(f"Received organic request: {folding_params}")
    request.app.state.validator._organic_queue.add(folding_params)

    return {"job_id": folding_params["job_id"]}
