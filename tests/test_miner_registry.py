import pytest
import folding 
from folding.rewards.miner_registry import MinerRegistry
from unittest.mock import patch
from typing import Dict, List 

miner_uids =[1,2]

@pytest.fixture
def create_reward_registry(miner_uids: List[int]= miner_uids):
    miner_uids=[1,2]
    reward_registry = MinerRegistry(miner_uids=miner_uids)
    return reward_registry

def test_class_instantiation(create_reward_registry):
    reward_registry = create_reward_registry

    # test class attributes return dictionaries.
    assert type(reward_registry.tasks) == dict
    assert type(reward_registry.registry) == dict

    # test miner_uids are in the registry   
    for index in reward_registry.registry:
        assert index in(miner_uids)



def test_add_results(create_reward_registry):
    reward_registry = create_reward_registry
    results: List[int] = [8]
    miner_uid: int = miner_uids[0]
    task: str = "SyntheticMDReward" 
    reward_registry.add_results(results=results, miner_uid=miner_uid, task=task)

    # assert 8 is in the registry for the appropriate miner 
    for score in reward_registry.registry[miner_uid][task]["results"]:
        assert score == 8


def test_compute_results(create_reward_registry):
    reward_registry = create_reward_registry
    miner_uid: int = miner_uids[0]
    task: str = "SyntheticMDReward"

    patch.object(reward_registry.registry[miner_uid][task]["results"].calculate_reward()).return_value(7)

    computed_results = reward_registry.compute_results(miner_uid=miner_uid, task=task)
    assert computed_results==[7]