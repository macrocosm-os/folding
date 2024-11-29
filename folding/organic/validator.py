import time
import copy
import random
import asyncio
import bittensor as bt
from folding.utils.logger import logger

from typing import Any, Literal, Union, Tuple

from folding.protocol import OrganicSynapse
from folding.base.neuron import BaseNeuron
from folding.utils.opemm_simulation_config import SimulationConfig

from atom.organic_scoring import OrganicScoringBase


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""

    def __init__(
        self,
        axon: bt.axon,
        trigger_frequency: Union[float, int],
        trigger: Literal["seconds", "steps"],
        validator: BaseNeuron,
        trigger_frequency_min: Union[float, int] = 5,
        trigger_scaling_factor: Union[float, int] = 5,
        synth_dataset=None,
    ):
        super().__init__(
            axon=axon,
            synth_dataset=synth_dataset,
            trigger_frequency=trigger_frequency,
            trigger=trigger,
            trigger_frequency_min=trigger_frequency_min,
            trigger_scaling_factor=trigger_scaling_factor,
        )

        # Self reference the validator object to have access to validator methods.
        self._validator: BaseNeuron = validator

        protected_args = ["ff", "water", "box"]
        simulation_args = list(
            SimulationConfig(ff="organic", water="organic", box="cube").to_dict().keys()
        )
        self.simulation_args = [
            arg for arg in simulation_args if arg not in protected_args
        ]

    async def _on_organic_entry(self, synapse: OrganicSynapse) -> bt.Synapse:
        """
        This is the entry point for the organic scoring task.
        It receives a synapse object from the axon and processes.
        """

        config: dict = synapse.get_simulation_params()
        self._organic_queue.add(config)
        logger.success(
            f"Query received: organic queue size = {self._organic_queue.size}"
        )

        # TODO: This is still False on the API side.... Why!???!
        synapse.is_processed = True
        return synapse

    async def start_loop(self):
        """
        The main loop for running the organic scoring task, either based on a time interval or steps.
        Calls the `sample` method to establish the sampling logic for the organic scoring task.
        """
        while not self._should_exit:
            if self._trigger == "steps":
                while self._step_counter < self._trigger_frequency:
                    await asyncio.sleep(0.1)

            try:
                logs = await self.forward()

                total_elapsed_time = logs.get("total_elapsed_time", 0)
                logger.info(
                    f"Organic scoring iteration completed in {total_elapsed_time:.2f} seconds."
                )

                logger.warning(
                    f"Sleeping for {self._validator.config.neuron.organic_trigger_frequency} seconds before next organic check."
                )
                await asyncio.sleep(
                    self._validator.config.neuron.organic_trigger_frequency
                )

            except Exception as e:
                logger.error(
                    f"Error occured during organic scoring iteration:\n{e}"
                )
                await asyncio.sleep(1)

    async def sample(self) -> dict[str, Any]:
        """Sample data from the organic queue or the synthetic dataset.

        Returns:
            dict[str, Any]: dict that contains all the attributes for creating a simulation object.
        """
        if not self._organic_queue.is_empty():
            # Choose organic sample based on the organic queue logic.
            sample = self._organic_queue.sample()
        elif self._synth_dataset is not None:
            # Choose if organic queue is empty, choose random sample from provided datasets.
            sample = random.choice(self._synth_dataset).sample()
        else:
            sample = None

        return sample

    async def forward(self) -> dict[str, Any]:
        """The forward method is responsible for sampling data from the organic queue,
        and adding it to the local database of the validator.
        """
        init_time = time.perf_counter()
        sample: dict[str, Any] = await self.sample()

        if sample is None:
            return {
                "total_elapsed_time": time.perf_counter() - init_time,
                "job_added": False,
            }

        sample_copy = copy.deepcopy(sample)

        job_event = {"system_kwargs": {}, "is_organic": True}
        for arg in self.simulation_args:
            if arg in sample:
                job_event["system_kwargs"][arg] = sample.pop(arg)

        # merge the keys from the sample to the job_event
        job_event.update(sample)

        # Add jobs to the sqlite database for the vali to process.
        job_added: bool = await self._validator.add_job(job_event=job_event)

        # If the job was unable to be added to the queue for any reason, add it back to the organic queue.
        if not job_added:
            logger.warning("adding previously sampled job back to organic queue.")
            self._organic_queue.add(sample_copy)
            return {
                "total_elapsed_time": time.perf_counter() - init_time,
                "job_added": False,
            }

        return {
            "total_elapsed_time": time.perf_counter() - init_time,
            "job_added": True,
        }

    def _blacklist_fn(self, synapse: OrganicSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self._validator.config.organic_whitelist:
            return True, ""
        return False, ""
