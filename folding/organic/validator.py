from typing import Tuple

from folding.protocol import OrganicSynapse
from atom.organic_scoring import OrganicScoringBase


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""
    
    ORGANIC_WHITELIST_HOTKEY = ''

    async def _on_organic_entry(self, synapse: OrganicSynapse) -> None:
        config: dict = synapse.get_simulation_params()
        self._organic_queue.add(config)

    def _blacklist_fn(self, synapse: OrganicSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey != self.ORGANIC_WHITELIST_HOTKEY:
            return True
        return False, ""
