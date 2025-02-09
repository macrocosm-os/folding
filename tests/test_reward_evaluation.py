import pytest
from unittest.mock import MagicMock
from folding.validators.protein import Protein
from folding.protocol import JobSubmissionSynapse
from folding.validators.reward import (
    evaluate_energies,
    process_valid_energies,
    get_energies,
)  # Replace `your_module` with actual module name


@pytest.fixture
def mock_protein():
    """Creates a mock Protein object."""
    protein = MagicMock(spec=Protein)
    protein.pdb_id = "mock_pdb"
    protein.pdb_location = "/mock/path"
    protein.pdb_directory = "/mock/directory"
    protein.system_config = {}
    protein.velm_array_pkl = "/mock/velm.pkl"
    return protein


@pytest.fixture
def mock_responses():
    """Creates mock JobSubmissionSynapse responses."""

    class MockDendrite:
        def __init__(self, status_code):
            self.status_code = status_code

    class MockAxon:
        def __init__(self, hotkey):
            self.hotkey = hotkey

    responses = []
    for i in range(5):
        resp = MagicMock(spec=JobSubmissionSynapse)
        resp.dendrite = MockDendrite(status_code=200)
        resp.axon = MockAxon(hotkey=f"hotkey_{i}")
        resp.miner_state = f"state_{i}"
        resp.miner_seed = f"seed_{i}"
        resp.md_output = f"output_{i}"
        responses.append(resp)
    return responses


@pytest.fixture
def mock_evaluation_registry():
    """Creates a mock evaluation registry with mock evaluators."""
    mock_registry = {}

    class MockEvaluator:
        def __init__(
            self, reported_energy, median_energy, ns_computed, checkpoint_path=None
        ):
            self.reported_energy = reported_energy
            self.median_energy = median_energy
            self.ns_computed = ns_computed
            self.checkpoint_path = checkpoint_path

        def evaluate(self):
            return True

        def get_reported_energy(self):
            return self.reported_energy

        def get_ns_computed(self):
            return self.ns_computed

        def validate(self):
            return self.median_energy, [self.median_energy], self.median_energy, "valid"

    # Populate mock registry with different energy values
    for i, energy in enumerate([10, 5, 15, 0, 20]):  # One invalid (0) energy
        mock_registry[f"job_type_{i}"] = lambda **kwargs: MockEvaluator(
            reported_energy=energy,
            median_energy=energy if energy > 0 else 0,  # Only nonzero median
            ns_computed=100 + i,
        )

    return mock_registry


def test_evaluate_energies(
    mock_protein, mock_responses, mock_evaluation_registry, monkeypatch
):
    """Test evaluation step with mock data."""
    monkeypatch.setattr(
        "folding.registries.evaluation_registry.EVALUATION_REGISTRY",
        mock_evaluation_registry,
    )

    uids = [1001, 1002, 1003, 1004, 1005]
    job_type = "job_type_0"

    results = evaluate_energies(mock_protein, mock_responses, uids, job_type)

    # Ensure correct number of results (ignoring invalid energy of 0)
    assert len(results) == 4
    assert all(res["reported_energy"] > 0 for res in results)


def test_process_valid_energies():
    """Test the processing of energies, ensuring correct validation and duplicates handling."""
    event = {
        "is_valid": {},
        "checked_energy": {},
        "reported_energy": {},
        "miner_energy": {},
        "reason": {},
        "is_run_valid_time": {},
        "ns_computed": {},
        "is_duplicate": {},
    }

    results = [
        {
            "uid": 1001,
            "reported_energy": 10,
            "evaluator": MagicMock(),
            "seed": "s1",
            "best_cpt": "",
            "process_time": 1.0,
        },
        {
            "uid": 1002,
            "reported_energy": 5,
            "evaluator": MagicMock(),
            "seed": "s2",
            "best_cpt": "",
            "process_time": 1.0,
        },
        {
            "uid": 1003,
            "reported_energy": 15,
            "evaluator": MagicMock(),
            "seed": "s3",
            "best_cpt": "",
            "process_time": 1.0,
        },
    ]

    # Mock evaluators' validate() method
    for res in results:
        res["evaluator"].validate.return_value = (
            res["reported_energy"],
            [res["reported_energy"]],
            res["reported_energy"],
            "valid",
        )
        res["evaluator"].get_ns_computed.return_value = 100

    processed_event = process_valid_energies(results, event)

    # Check that the processed event has valid energies
    assert processed_event["is_valid"][1001] is True
    assert processed_event["is_valid"][1002] is True
    assert processed_event["is_valid"][1003] is True


def test_get_energies(
    mock_protein, mock_responses, mock_evaluation_registry, monkeypatch
):
    """Test full pipeline execution and verify final energy output."""
    monkeypatch.setattr(
        "folding.registries.evaluation_registry.EVALUATION_REGISTRY",
        mock_evaluation_registry,
    )

    uids = [1001, 1002, 1003, 1004, 1005]
    job_type = "job_type_0"

    energies, event = get_energies(mock_protein, mock_responses, uids, job_type)

    # Ensure correct number of energies returned in original UID order
    assert len(energies) == len(uids)

    # Verify energies are sorted according to UIDs, not sorted energy values
    expected_energies = [10, 5, 15, 0, 20]  # Corresponding to mock_evaluation_registry
    assert list(energies) == expected_energies

    # Ensure valid and duplicate flags are correctly updated
    assert event["is_valid"][1001] is True
    assert event["is_valid"][1002] is True
    assert event["is_valid"][1003] is True
    assert event["is_valid"][1004] is False  # UID 1004 had reported energy 0
    assert event["is_valid"][1005] is True
