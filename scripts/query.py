import asyncio
import bittensor as bt
from folding.protocol import OrganicSynapse


def query(validator_uid: int, params: dict):
    metagraph = bt.metagraph(netuid=141, network="test")
    axons = [metagraph.axons[validator_uid]]

    wallet = bt.wallet(name="folding-testnet", hotkey="m4")
    print(wallet.hotkey.ss58_address)
    dendrite = bt.dendrite(wallet=wallet)

    validator_response: OrganicSynapse = dendrite.query(
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

    validator_uid = 81  # Replace with the actual validator UID

    # Run the async function using asyncio.run()
    # response = asyncio.run(query(validator_uid, params))
    response = query(validator_uid, params)
