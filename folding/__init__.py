from .protocol import JobSubmissionSynapse
from .validators.protein import Protein

__version__ = "1.0.0"
version_split = __version__.split(".")
__spec_version__ = (
    (10000 * int(version_split[0]))
    + (100 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

__OPENMM_VERSION_TAG__ = "8.1.2"
