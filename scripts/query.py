import asyncio
import bittensor as bt
from folding.protocol import OrganicSynapse


async def query(validator_uid: int, params: dict):
    metagraph = bt.metagraph(netuid=141, network="testnet")
    axons = [metagraph.axons[validator_uid]]

    validator_response: OrganicSynapse = await bt.dendrite.forward(
        axons=axons,
        synapse=OrganicSynapse(**params),
        timeout=10,
        deserialize=False,
    )

    return validator_response


if __name__ == "__main__":
    params = {
        "pdb_id": "1AKI",
        "ff": "amber",
        "water": "tip3p",
        "box": "cubic",
        "seed": 0,
        "temperature": 300.0,
        "friction": 1.0,
    }

    validator_uid = 101  # Replace with the actual validator UID

    # Run the async function using asyncio.run()
    response = asyncio.run(query(validator_uid, params))
