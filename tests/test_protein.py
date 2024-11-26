import pytest

import os
from collections import defaultdict
from pathlib import Path
from folding.utils.ops import gro_hash
from folding.validators.protein import Protein

ROOT_PATH = Path(__file__).parent


results = {
    "1fjs": defaultdict(
        int, {"REMARK": 356, "ATOM": 2236, "HETATM": 239, "CONECT": 98}
    ),
    "1ubq": defaultdict(int, {"REMARK": 260, "ATOM": 602, "HETATM": 58}),
}


@pytest.mark.parametrize(
    "pdb_file, expected",
    [
        (os.path.join(ROOT_PATH, "fixtures/pdb_files", f"{pdb}.pdb"), result)
        for pdb, result in results.items()
    ],
)
def test_pdb_complexity(pdb_file, expected):
    record_counts = Protein._get_pdb_complexity(pdb_file)
    assert record_counts == expected, "record counts do not match expected values"
