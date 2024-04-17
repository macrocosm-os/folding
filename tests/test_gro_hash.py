import pytest
from folding.utils.ops import gro_hash


MINER_GRO_FILE = "/root/folding/tests/fixtures/gro_files/md_0_1.gro"
VALIDATOR_GRO_FILE = "/root/folding/tests/fixtures/gro_files/em.gro"


def test_gro_hash():
    validator_hash = gro_hash(gro_path=VALIDATOR_GRO_FILE)
    miner_hash = gro_hash(gro_path=MINER_GRO_FILE)

    assert (
        validator_hash == miner_hash
    ), "validator gro file hash DNE miner gro file hash"
