from folding.utils.constants import STARTING_CREDIBILITY
from folding.registries.miner_registry import MinerRegistry
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


def setup_registry():
    """
    Helper function to create and initialize a MinerRegistry with test miners.

    Returns:
        MinerRegistry: A registry initialized with miner UIDs 1, 2, and 3.
    """
    miner_uids = [1, 2, 3]
    miner_registry = MinerRegistry(miner_uids=miner_uids)
    return miner_registry


def test_miner_registry():
    """
    Test that the MinerRegistry correctly initializes with tasks from the EVALUATION_REGISTRY.

    Verifies that the tasks in the MinerRegistry match the keys in the EVALUATION_REGISTRY.
    """
    miner_registry = setup_registry()
    assert list(miner_registry.tasks) == list(EVALUATION_REGISTRY.keys())


def test_miner_credibilities():
    """
    Test the credibility update mechanism in the MinerRegistry.

    This test verifies:
    1. Credibilities are correctly added to a miner's task metrics
    2. Positive credibilities increase a miner's task credibility
    3. Negative credibilities decrease a miner's task credibility
    4. After negative credibilities, the final credibility is below the starting value
    """
    miner_registry = setup_registry()
    task_name = miner_registry.tasks[0]  # select the first task for example.

    miner_uid = 1

    # Accepting a batch of credibility data.
    credibility_batch = [1.0, 1.0, 1.0]
    miner_registry.add_credibilities(
        miner_uid=miner_uid, task=task_name, credibilities=credibility_batch
    )

    assert miner_registry.registry[miner_uid].tasks[task_name].credibilities == [
        credibility_batch
    ]

    previous_credibility = (
        miner_registry.registry[miner_uid].tasks[task_name].credibility
    )
    miner_registry.update_credibility(miner_uid=miner_uid, task=task_name)

    assert (
        miner_registry.registry[miner_uid].tasks[task_name].credibility
        > previous_credibility
    )

    credibility_batch = [0.0, 0.0, 0.0]
    miner_registry.add_credibilities(
        miner_uid=miner_uid, task=task_name, credibilities=credibility_batch
    )

    previous_credibility = (
        miner_registry.registry[miner_uid].tasks[task_name].credibility
    )
    miner_registry.update_credibility(miner_uid=miner_uid, task=task_name)

    assert (
        miner_registry.registry[miner_uid].tasks[task_name].credibility
        < previous_credibility
    )
    assert (
        miner_registry.registry[miner_uid].tasks[task_name].credibility
        < STARTING_CREDIBILITY
    )
