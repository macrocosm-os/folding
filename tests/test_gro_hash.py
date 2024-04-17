import pytest

import os
from pathlib import Path
from folding.utils.ops import gro_hash

ROOT_PATH = Path(__file__).parent

MINER_GRO_FILE = os.path.join(ROOT_PATH, "fixtures/gro_files/md_0_1.gro")
MINER_GRO_FILE_CORRUPT = os.path.join(
    ROOT_PATH, "fixtures/gro_files/md_0_1_altered.gro"
)
VALIDATOR_GRO_FILE = os.path.join(ROOT_PATH, "fixtures/gro_files/em.gro")


def test_gro_hash():
    validator_hash = gro_hash(gro_path=VALIDATOR_GRO_FILE)
    miner_hash = gro_hash(gro_path=MINER_GRO_FILE)
    miner_hash_corrupt = gro_hash(gro_path=MINER_GRO_FILE_CORRUPT)

    assert (
        validator_hash == miner_hash
    ), "validator gro file hash DNE miner gro file hash"

    assert (
        validator_hash != miner_hash_corrupt
    ), "validator gro hash and corrupt miner gro hash are the same"
