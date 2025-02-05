from folding.utils.constants import STARTING_CREDIBILITY
from folding.registries.miner_registry import MinerRegistry
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


def setup_registry():
    miner_uids = [1, 2, 3]
    miner_registry = MinerRegistry(miner_uids=miner_uids)
    return miner_registry


def test_miner_registry():
    miner_registry = setup_registry()
    assert list(miner_registry.tasks) == list(EVALUATION_REGISTRY.keys())


def test_miner_credibilities():
    miner_registry = setup_registry()
    task_name = miner_registry.tasks[0]  # select the first task for example.

    miner_uid = 1

    # Accepting a batch of credibility data.
    credibility_batch = [1.0, 1.0, 1.0]
    miner_registry.add_credibilities(
        miner_uid=miner_uid, task=task_name, credibilities=credibility_batch
    )

    assert miner_registry.registry[miner_uid][task_name]["credibilities"] == [
        credibility_batch
    ]

    previous_credibility = miner_registry.registry[miner_uid][task_name]["credibility"]
    miner_registry.update_credibility(miner_uid=miner_uid, task=task_name)

    assert (
        miner_registry.registry[miner_uid][task_name]["credibility"]
        > previous_credibility
    )

    credibility_batch = [0.0, 0.0, 0.0]
    miner_registry.add_credibilities(
        miner_uid=miner_uid, task=task_name, credibilities=credibility_batch
    )

    previous_credibility = miner_registry.registry[miner_uid][task_name]["credibility"]
    miner_registry.update_credibility(miner_uid=miner_uid, task=task_name)

    assert (
        miner_registry.registry[miner_uid][task_name]["credibility"]
        < previous_credibility
    )
    assert (
        miner_registry.registry[miner_uid][task_name]["credibility"]
        < STARTING_CREDIBILITY
    )
