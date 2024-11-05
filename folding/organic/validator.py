from typing import Any, override

from folding.protocol import OrganicSynapse
from atom.organic_scoring import OrganicScoringBase

from folding.utils.opemm_simulation_config import SimulationConfig


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""

    @override
    async def _on_organic_entry(self, synapse: OrganicSynapse) -> None:
        simulation_config: SimulationConfig = synapse.deserialize()
        self._organic_queue.add(simulation_config.to_dict())
