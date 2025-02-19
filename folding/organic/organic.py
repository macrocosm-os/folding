import uuid
from fastapi import APIRouter, Request, Depends, HTTPException
from atom.epistula.epistula import Epistula, VerifySignatureRequest
from folding_api.schemas import FoldingSchema
from folding.utils.logging import logger

epistula = Epistula()

router = APIRouter()


@router.get("/organic")
async def organic(
    request: Request,
    job: FoldingSchema,
    epistula_headers: VerifySignatureRequest = Depends(epistula.verify_signature),
):
    """
    This endpoint is used to receive organic requests. Returns success message with the job id.
    Args:
        request: Request
        job: FoldingSchema
        epistula_headers: VerifySignatureRequest
    Returns:
        dict[str, str]: dict with the job id.
    """

    sender_hotkey = epistula_headers.signed_by

    if sender_hotkey not in request.app.state.config.neuron.organic_whitelist:
        logger.warning(
            f"Received organic request from {sender_hotkey}, but {sender_hotkey} is not in the whitelist."
        )
        raise HTTPException(
            status_code=403, detail="Forbidden, sender not in whitelist."
        )

    folding_params = job.folding_params
    folding_params["job_id"] = str(uuid.uuid4())
    logger.info(f"Received organic request: {folding_params}")
    request.app.state.validator._organic_queue.add(folding_params)

    return {"job_id": folding_params["job_id"]}
