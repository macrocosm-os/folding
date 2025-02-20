from typing import List
from collections import defaultdict

from folding_api.schemas import FoldingSchema, FoldingReturn
from folding.protocol import OrganicSynapse
from folding_api.vars import subtensor_service, validator_registry


async def query_validators(schema: FoldingSchema) -> FoldingReturn:
    """
    Query validators with the given parameters and return a streaming
    """
    # Get the UIDs (and axons) to query by looking at the top validators
    # check if uids are in registry, if all of them are not default to random choice
    if schema.api_parameters["validator_uids"] is not None and all(
        uid in validator_registry.validators
        for uid in schema.api_parameters["validator_uids"]
    ):
        uids = schema.api_parameters["validator_uids"]
        addresses = [validator_registry.validators[uid].address for uid in uids]
    else:
        available_uids = validator_registry.get_available_validators()
        selected_uids = available_uids[
            : schema.api_parameters["num_validators_to_sample"]
        ]
        addresses = [
            validator_registry.validators[uid].address for uid in selected_uids
        ]

    validator_responses: List[OrganicSynapse] = await subtensor_service.dendrite(
        addresses=addresses,
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
