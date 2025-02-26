import time
import copy
import asyncio
import uuid
import bittensor as bt
from folding.utils.logger import logger

from typing import Any, Literal, Union, Optional, cast

from folding.base.neuron import BaseNeuron
from folding.utils.opemm_simulation_config import SimulationConfig

from folding.base.organic_scoring_base import OrganicScoringBase
from atom.organic_scoring.organic_queue import OrganicQueue


class OrganicValidator(OrganicScoringBase):
    """Validator class to handle organic entries."""

    def __init__(
        self,
        trigger_frequency: Union[float, int],
        trigger: Literal["seconds", "steps"],
        validator: BaseNeuron,
    ):
        super().__init__(
            trigger_frequency=trigger_frequency,
            trigger=trigger,
            organic_queue=OrganicQueue(),  # Initialize with concrete type
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

    async def start_loop(self):
        """
        The main loop for running the organic scoring task, either based on a time interval or steps.
        Calls the `sample` method to establish the sampling logic for the organic scoring task.
        """
        while True:
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
                logger.error(f"Error occured during organic scoring iteration:\n{e}")
                await asyncio.sleep(1)

    async def forward(self) -> dict[str, Any]:
        """The forward method is responsible for sampling data from the organic queue,
        and adding it to the local database of the validator.
        """
        init_time = time.perf_counter()
        sample: Optional[dict[str, Any]] = await self.sample()

        if sample is None:
            return {
                "total_elapsed_time": time.perf_counter() - init_time,
                "job_added": False,
            }

        sample_copy = copy.deepcopy(sample)

        # TODO: Need to set the job type programmatically.
        job_event = {
            "system_kwargs": {},
            "is_organic": True,
            "job_type": "OrganicMD",
            "job_id": str(uuid.uuid4()),
            "epsilon": 0.5,
        }
        for arg in self.simulation_args:
            if arg in sample:
                job_event["system_kwargs"][arg] = sample.pop(arg)

        # merge the keys from the sample to the job_event
        job_event.update(sample)

        # Add jobs to the sqlite database for the vali to process.
        # Cast validator to Any since add_job is added dynamically
        job_added: bool = await cast(Any, self._validator).add_job(job_event=job_event)

        # If the job was unable to be added to the queue for any reason, add it back to the organic queue.
        if not job_added and self._organic_queue is not None:
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

    async def sample(self) -> Optional[dict[str, Any]]:
        """Sample data from the organic queue.

        Returns:
            Optional[dict[str, Any]]: dict that contains all the attributes for creating a simulation object,
            or None if no sample is available.
        """
        if self._organic_queue and not self._organic_queue.is_empty():
            # Choose organic sample based on the organic queue logic.
            return self._organic_queue.sample()
        return None
