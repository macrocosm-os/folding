import json
from folding_api.schemas import FoldingParams
import requests
from folding_api.vars import epistula, subtensor_service


async def make_request(
    address: str, folding_params: FoldingParams
) -> requests.Response:
    body_bytes = json.dumps(
        folding_params.model_dump(), default=str, sort_keys=True
    ).encode("utf-8")
    headers = epistula.generate_header(subtensor_service.wallet.hotkey, body_bytes)
    response = requests.post(f"{address}/organic", data=body_bytes, headers=headers)
    return response
