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
