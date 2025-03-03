import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal, Union

import bittensor as bt

from atom.organic_scoring.organic_queue import OrganicQueue, OrganicQueueBase


class OrganicScoringBase(ABC):
    def __init__(
        self,
        trigger_frequency: Union[float, int],
        trigger: Literal["seconds", "steps"],
        organic_queue: OrganicQueueBase = None,
    ):
        """Runs the organic scoring task in separate threads.

        Args:
            axon: The axon to use, must be started and served.
            trigger_frequency: The frequency to trigger the organic scoring reward step.
            trigger: The trigger type, available values: "seconds", "steps".
                In case of "seconds" the `trigger_frequency` is the number of seconds to wait between each step.
                In case of "steps" the `trigger_frequency` is the number of steps to wait between each step. The
                `increment_step` method should be called to increment the step counter.
            organic_queue: The organic queue to use, must be inherited from `organic_queue.OrganicQueueBase`.
                Defaults to `organic_queue.OrganicQueue`.

        Override the following methods:
            - `forward`: Method to establish the sampling logic for the organic scoring task.
        """
        self._trigger_frequency = trigger_frequency
        self._trigger = trigger

        self._organic_queue = organic_queue
        if self._organic_queue is None:
            self._organic_queue = OrganicQueue()

        self._step_counter = 0
        self._step_lock = asyncio.Lock()

    def increment_step(self):
        """Increment the step counter if the trigger is set to `steps`."""
        with self._step_lock:
            if self._trigger == "steps":
                self._step_counter += 1

    @abstractmethod
    async def forward(self) -> dict[str, Any]:
        """
        Method to establish the sampling logic for the organic scoring task.
        Sample data from the organic queue or the synthetic dataset (if available).

        Expected to return a dictionary with information from the sampling method.
        If the trigger is based on seconds, the dictionary should contain the key "total_elapsed_time".
        """
        ...

    async def start_loop(self):
        """The main loop for running the organic scoring task, either based on a time interval or steps.
        Calls the `sample` method to establish the sampling logic for the organic scoring task.
        """
        while True:
            try:
                logs = await self.forward()

                total_elapsed_time = logs.get("total_elapsed_time", 0)
                await self.wait_until_next(timer_elapsed=total_elapsed_time)

            except Exception as e:
                bt.logging.error(
                    f"Error occured during organic scoring iteration:\n{e}"
                )
                await asyncio.sleep(1)

    async def wait_until_next(self, timer_elapsed: float = 0):
        """Wait until next iteration based on the trigger type and elapsed time.

        Args:
            timer_elapsed: The time elapsed during the current iteration of the processing loop. This is used
                to calculate the remaining sleep duration when the trigger is based on seconds.
        """
        if self._trigger == "seconds":
            # Adjust the sleep duration based on elapsed time
            sleep_duration = max(self._trigger_frequency - timer_elapsed, 0)
            await asyncio.sleep(sleep_duration)
        elif self._trigger == "steps":
            # Wait until enough steps have accumulated
            while True:
                if self._step_counter >= self._trigger_frequency:
                    self._step_counter -= self._trigger_frequency
                else:
                    await asyncio.sleep(1)
