import pytest
import torch
from folding.base.reward import BatchRewardInput, BatchRewardOutput
from folding.store import MockJob
from folding.rewards.folding_reward import FoldingReward


@pytest.fixture
def mock_job():
    mock_job = MockJob()
    mock_job.best_hotkey = "hotkey1"
    mock_job.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
    mock_job.best_loss = 0.5
    mock_job.event = {"is_organic": False}
    return mock_job


@pytest.fixture
def folding_reward():
    return FoldingReward()


@pytest.mark.asyncio
async def test_get_rewards_best_hotkey_not_in_set(folding_reward, mock_job):
    # Modify job to have best_hotkey not in hotkeys
    mock_job.best_hotkey = "non_existent_hotkey"

    data = BatchRewardInput(
        energies=torch.tensor([0.1, 0.2, 0.3]), top_reward=0.8, job=mock_job
    )
    rewards = torch.zeros(3)

    result = await folding_reward.get_rewards(data, rewards)

    assert torch.all(result.rewards == 0)
    assert result.extra_info["name"] == folding_reward.name()
    assert result.extra_info["top_reward"] == 0.8


@pytest.mark.asyncio
async def test_get_rewards_no_replies(folding_reward, mock_job):
    data = BatchRewardInput(energies=torch.zeros(3), top_reward=0.8, job=mock_job)
    rewards = torch.zeros(3)

    result = await folding_reward.get_rewards(data, rewards)

    # Best hotkey (index 0) should get reward of 1, others 0
    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.all(result.rewards == expected)


@pytest.mark.asyncio
async def test_get_rewards_only_best_replies(folding_reward, mock_job):
    energies = torch.zeros(3)
    energies[0] = 0.5  # Only best hotkey replies

    data = BatchRewardInput(energies=energies, top_reward=0.8, job=mock_job)
    rewards = torch.zeros(3)

    result = await folding_reward.get_rewards(data, rewards)

    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.all(result.rewards == expected)


@pytest.mark.asyncio
async def test_get_rewards_multiple_best_values(folding_reward, mock_job):
    energies = torch.tensor([0.5, 0.5, 0.3])  # Two miners with best loss

    data = BatchRewardInput(energies=energies, top_reward=0.8, job=mock_job)
    rewards = torch.zeros(3)

    result = await folding_reward.get_rewards(data, rewards)

    # Both miners with best loss should get top_reward
    assert result.rewards[0] == 0.8
    assert result.rewards[1] == 0.8
    assert result.rewards[2] < 0.8  # Third miner should get lower reward


@pytest.mark.asyncio
async def test_get_rewards_distribution(folding_reward, mock_job):
    energies = torch.tensor([0.5, 0.6, 0.7])  # Different energy values

    data = BatchRewardInput(energies=energies, top_reward=0.8, job=mock_job)
    rewards = torch.zeros(3)

    result = await folding_reward.get_rewards(data, rewards)

    # Check that rewards are properly distributed
    assert result.rewards[0] == 0.8  # Best miner gets top_reward
    assert result.rewards[1] > result.rewards[2]  # Better performers get higher rewards
    assert torch.all(result.rewards >= 0)  # No negative rewards
    assert torch.all(result.rewards <= 1)  # No rewards above 1


@pytest.mark.asyncio
async def test_calculate_final_reward_organic(folding_reward):
    rewards = torch.tensor([0.8, 0.5, 0.3])

    job = MockJob()
    job.best_hotkey = "hotkey1"
    job.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
    job.best_loss = 0.5
    job.event = {"is_organic": True}
    result = await folding_reward.calculate_final_reward(rewards, job)

    # Organic multiplier should be 10.0
    expected = rewards * 10.0
    assert torch.all(result == expected)


@pytest.mark.asyncio
async def test_calculate_final_reward_non_organic(folding_reward):
    rewards = torch.tensor([0.8, 0.5, 0.3])
    job = MockJob()
    job.best_hotkey = "hotkey1"
    job.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
    job.best_loss = 0.5
    job.event = {"is_organic": False}

    result = await folding_reward.calculate_final_reward(rewards, job)

    # Non-organic multiplier should be 1.0
    expected = rewards * 1.0
    assert torch.all(result == expected)


@pytest.mark.asyncio
async def test_full_reward_pipeline(folding_reward, mock_job):
    data = BatchRewardInput(
        energies=torch.tensor([0.5, 0.6, 0.7]), top_reward=0.8, job=mock_job
    )

    result = await folding_reward.apply(data)

    assert isinstance(result.rewards, torch.Tensor)
    assert result.reward_name == folding_reward.name()
    assert isinstance(result.batch_time, float)
    assert result.batch_time > 0
    assert result.extra_info is not None
