import os
import openmm.app as app


class LastTwoCheckpointsReporter(app.CheckpointReporter):
    def __init__(self, file_prefix, reportInterval):
        super().__init__(file_prefix + "_1.cpt", reportInterval)
        self.file_prefix = file_prefix
        self.reportInterval = reportInterval

    def report(self, simulation, state):
        # Create a new checkpoint
        current_checkpoint = f"{self.file_prefix}.cpt"
        if os.path.exists(current_checkpoint):
            os.rename(current_checkpoint, f"{self.file_prefix}_old.cpt")
        simulation.saveCheckpoint(current_checkpoint)

    def describeNextReport(self, simulation):
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, False, False, False, False, False)


class SequentialCheckpointReporter(app.CheckpointReporter):
    """A Reporter that saves checkpoints with sequential numbering.

    This reporter saves a new checkpoint file at regular intervals, naming each file
    with a sequential number that increases with each checkpoint.
    """

    def __init__(self, file_prefix, reportInterval, checkpoint_counter=0):
        """Create a SequentialCheckpointReporter.

        Parameters
        ----------
        file_prefix : str
            The prefix for checkpoint files. Each checkpoint will be saved as
            {file_prefix}_{counter}.cpt where counter is an incrementing number.
        reportInterval : int
            The interval (in time steps) at which to save checkpoints
        checkpoint_counter : int, optional
            The starting value for the checkpoint counter. Default is 1.
        """
        self.file_prefix = file_prefix
        self.reportInterval = reportInterval
        self.checkpoint_counter = checkpoint_counter
        super().__init__(file_prefix + f"{self.checkpoint_counter}.cpt", reportInterval)

    def report(self, simulation, state):
        """Generate a checkpoint file with a sequential number.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a checkpoint for
        state : State
            The current state of the simulation
        """
        checkpoint_path = f"{self.file_prefix}_{self.checkpoint_counter}.cpt"
        simulation.saveCheckpoint(checkpoint_path)
        self.checkpoint_counter += 1

    def describeNextReport(self, simulation):
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, False, False, False, False, False)


class ExitFileReporter(object):
    def __init__(self, filename, reportInterval, file_prefix):
        self.filename = filename
        self.reportInterval = reportInterval
        self.file_prefix = file_prefix

    def describeNextReport(self, simulation):
        steps_left = simulation.currentStep % self.reportInterval
        return (steps_left, False, False, False, False)

    def report(self, simulation, state):
        if os.path.exists(self.filename):
            with open(f"{self.file_prefix}.cpt", "wb") as f:
                f.write(simulation.context.createCheckpoint())
            raise Exception("Simulation stopped")

    def finalize(self):
        pass
