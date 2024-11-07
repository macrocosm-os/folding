from typing import Any
from folding.protocol import OrganicSynapse
from atom.organic_scoring import OrganicScoringBase


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""

    async def _on_organic_entry(self, synapse: OrganicSynapse) -> None:
        config: dict = synapse.get_simulation_params()
        self._organic_queue.add(config)
