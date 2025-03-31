import io
import os
import time
import openmm.app as app
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
import numpy as np




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
        checkpoint_path = f"{self.file_prefix}{self.checkpoint_counter}.cpt"
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


class ProteinStructureReporter(app.StateDataReporter):
    def __init__(
        self, file, reportInterval, reference_pdb, **kwargs
    ):
        super().__init__(file, reportInterval, **kwargs)
        self.reference_universe = mda.Universe(reference_pdb)
        self.positions_history = []  # Store positions for RMSF calculation

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(simulation)
            headers = self._constructHeaders()
            if not self._append:
                print(
                    '#"%s"' % ('"' + self._separator + '"').join(headers),
                    file=self._out,
                )
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Store current positions for RMSF calculation
        universe = self.create_mda_universe(simulation)
        self.positions_history.append(universe.select_atoms("backbone").positions.copy())

        # Query for the values
        values = self._constructReportValues(simulation, state)

        # Write the values.
        print(self._separator.join(str(v) for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        rmsd = self._calculate_rmsd(self.create_mda_universe(simulation))
        rmsf = self._calculate_rmsf()
        values.extend([rmsd, rmsf])
        return values

    def _calculate_rmsd(self, universe):
        """Calculate RMSD between current and reference positions.

        Args:
            positions (np.ndarray): Current positions

        Returns:
            float: RMSD
        """
        current_positions = universe.select_atoms("backbone").positions.copy()
        reference_positions = self.reference_universe.select_atoms("backbone").positions.copy()
        rmsd_measure = rmsd(current_positions, reference_positions, center=True)
        return rmsd_measure

    def _calculate_rmsf(self):
        """Calculate RMSF (Root Mean Square Fluctuation) over time.

        Returns:
            float: RMSF value in nanometers
        """
        if len(self.positions_history) < 2:
            return 0.0
            
        # Convert positions history to numpy array
        positions_array = np.array(self.positions_history)
        
        # Calculate mean position for each atom
        mean_positions = np.mean(positions_array, axis=0)
        
        # Calculate RMSF
        squared_diff = np.square(positions_array - mean_positions)
        rmsf = np.sqrt(np.mean(squared_diff))
        
        # Keep only the last 1000 frames to prevent memory issues
        if len(self.positions_history) > 1000:
            self.positions_history = self.positions_history[-1000:]
            
        return rmsf

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        headers.extend(["RMSD", "RMSF"])
        return headers

    def create_mda_universe(self,simulation):
        """
        Create an MDAnalysis Universe from an OpenMM simulation object.
        
        Args:
            simulation (openmm.app.Simulation): The OpenMM simulation object
            
        Returns:
            mda.Universe: An MDAnalysis Universe containing the current state of the simulation
        """
        # Get the current state
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        
        # Get the topology
        topology = simulation.topology
        
        # Create a PDB string from the current state
        pdb_string = io.StringIO()
        app.PDBFile.writeFile(topology, positions, pdb_string)
        pdb_string.seek(0)
        
        # Create MDAnalysis Universe from the PDB string
        universe = mda.Universe(pdb_string, format='pdb')
        
        return universe
   