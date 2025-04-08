import json
from typing import Optional, Dict, Any

from fastapi import UploadFile, File
from folding_api.schemas import FoldingParams
import requests
from folding_api.vars import epistula, subtensor_service, bt_config


async def make_request(
    address: str,
    folding_params: FoldingParams,
    pdb_file: Optional[UploadFile] = File(None),
) -> requests.Response:
    body_bytes = json.dumps(
        folding_params.model_dump(), default=str, sort_keys=True
    ).encode("utf-8")
    headers = epistula.generate_header(subtensor_service.wallet.hotkey, body_bytes)

    if pdb_file is not None:
        files: Dict[str, Any] = {
            "pdb_file": (pdb_file.filename, pdb_file.file, pdb_file.content_type)
        }
        # Remove content-type from headers as it will be set automatically for multipart form data
        headers.pop("Content-Type", None)
        return requests.post(
            f"{address}/organic/upload",
            data={"query": body_bytes.decode("utf-8")},
            headers=headers,
            files=files,
        )
    else:
        return requests.post(f"{address}/organic", data=body_bytes, headers=headers)


def response_to_dict(response) -> list[dict]:
    response = response.json()["results"][0]
    if "error" in response.keys():
        raise ValueError(f"Failed to get all PDBs: {response['error']}")
    elif "values" not in response.keys():
        return [{"error": response["error"]}]
    columns = response["columns"]
    values = response["values"]
    data = [dict(zip(columns, row)) for row in values]
    return data


def query_gjp(query: str) -> list[dict]:
    response = requests.get(
        f"http://{bt_config.gjp_address}/db/query", params={"q": query}
    )
    return response_to_dict(response)
