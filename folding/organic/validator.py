import time
import random
import bittensor as bt 
from typing import Any, Literal, Union, Tuple

from folding.protocol import OrganicSynapse
from folding.base.validator import BaseValidatorNeuron

from atom.organic_scoring import OrganicScoringBase


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""

    def __init__(
        self,
        axon: bt.axon,
        trigger_frequency: Union[float, int],
        trigger: Literal["seconds", "steps"],
        validator: BaseValidatorNeuron,
        trigger_frequency_min: Union[float, int] = 5,
        trigger_scaling_factor: Union[float, int] = 5,
        synth_dataset = None
    ):
    
        super().__init__(
            axon=axon,
            synth_dataset=synth_dataset,
            trigger_frequency=trigger_frequency,
            trigger=trigger,
            trigger_frequency_min=trigger_frequency_min,
            trigger_scaling_factor=trigger_scaling_factor,
        )

        self._validator = validator

    async def _on_organic_entry(self, synapse: OrganicSynapse) -> None:
        config: dict = synapse.get_simulation_params()
        self._organic_queue.add(config)

    async def sample(self) -> dict[str, Any]:
        if not self._organic_queue.is_empty():
            # Choose organic sample based on the organic queue logic.
            sample = self._organic_queue.sample()
        elif self._synth_dataset is not None:
            # Choose if organic queue is empty, choose random sample from provided datasets.
            sample = random.choice(self._synth_dataset).sample()
        else:
            # Return empty dictionary if no samples are available.
            sample = {}

        return sample

    async def forward(self) -> dict[str, Any]:
        init_time = time.perf_counter()

        sample = await self.sample()

        if not sample:
            return {"total_elapsed_time": time.perf_counter() - init_time}

        return {
            "total_elapsed_time": time.perf_counter() - init_time
        }

    def _blacklist_fn(self, synapse: OrganicSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey != self.ORGANIC_WHITELIST_HOTKEY:
            return True
        return False, ""
