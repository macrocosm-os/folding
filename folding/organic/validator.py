from typing import Any, override

from folding.protocol import OrganicSynapse
from atom.organic_scoring import OrganicScoringBase


class OrganicValidator(OrganicScoringBase):
    @override
    async def _on_organic_entry(self, synapse: OrganicSynapse) -> None:
        synapse
        self._organic_queue.add(synapse)
