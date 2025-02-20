from folding_api.schemas import FoldingParams
import requests
from folding_api.vars import epistula, subtensor_service


async def make_request(address: str, folding_params: FoldingParams):
    body_bytes = epistula.create_message_body(folding_params.model_dump())
    headers = epistula.generate_header(subtensor_service.wallet.hotkey, body_bytes)
    response = requests.post(
        f"{address}/organic", json=folding_params.model_dump(), headers=headers
    )
    return response.json()
