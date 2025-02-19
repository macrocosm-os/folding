from typing import List
from collections import defaultdict

from folding_api.schemas import FoldingSchema, FoldingReturn
from folding_api.utils import get_validator_data
from folding.protocol import OrganicSynapse
from folding_api.vars import subtensor_service


async def query_validators(schema: FoldingSchema) -> FoldingReturn:
    """
    Query validators with the given parameters and return a streaming
    """

    metagraph = subtensor_service.metagraph

    # Get the UIDs (and axons) to query by looking at the top validators
    if schema.api_parameters["validator_uids"] is not None:
        uids = schema.api_parameters["validator_uids"]
        axons = [metagraph.axons[uid] for uid in uids]
    else:
        sorted_validators = get_validator_data(metagraph=metagraph)
        validators = dict(
            list(sorted_validators.items())[
                : schema.api_parameters["num_validators_to_sample"]
            ]
        )
        axons = [metagraph.axons[v["uid"]] for v in validators]

    validator_responses: List[OrganicSynapse] = await subtensor_service.dendrite(
        axons=axons,
        synapse=OrganicSynapse(**schema.folding_params),
        timeout=schema.timeout,
        deserialize=False,
    )

    response_information = defaultdict(list)
    for resp in validator_responses:
        if resp.axon is not None:
            response_information["hotkeys"].append(resp.axon.hotkey)
            response_information["status_codes"].append(resp.axon.status_code)
        else:
            response_information["hotkeys"].append(None)
            response_information["status_codes"].append(None)

    return FoldingReturn(**response_information)
