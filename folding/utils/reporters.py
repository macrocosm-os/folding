import os
import time
import openmm.app as app
import numpy as np
from openmm import unit
import MDAnalysis as mda



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


class RMSDStateDataReporter(app.StateDataReporter):
    def __init__(
        self, file, reportInterval, reference_positions, protein_atoms, **kwargs
    ):
        super().__init__(file, reportInterval, **kwargs)
        self.reference_positions = reference_positions
        self.protein_atoms = protein_atoms
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
        current_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions_history.append(current_positions[self.protein_atoms])

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
        rmsd = self._calculate_rmsd(
            simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        )
        rmsf = self._calculate_rmsf()
        values.extend([rmsd, rmsf])
        return values

    def _calculate_rmsd(self, positions):
        """Calculate RMSD between current and reference positions.

        Args:
            positions (np.ndarray): Current positions

        Returns:
            float: RMSD value in nanometers
        """
        #select positions of non-water atoms
        positions = positions[self.protein_atoms]
        # Center the structures
        ref_centered = self.reference_positions - np.mean(
            self.reference_positions, axis=0
        )
        pos_centered = positions - np.mean(positions, axis=0)

        # Calculate RMSD
        diff = ref_centered.astype(np.float32) - pos_centered.astype(np.float32)
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

        return rmsd

    def _calculate_rmsf(self):
        """Calculate RMSF (Root Mean Square Fluctuation) over time.

        Returns:
            float: RMSF value in nanometers
        """
        if len(self.positions_history) < 2:
            return 0.0

        # Convert history to numpy array
        positions_array = np.array(self.positions_history)
        
        # Calculate mean position for each atom
        mean_positions = np.mean(positions_array, axis=0)
        
        # Calculate fluctuations from mean
        fluctuations = positions_array - mean_positions
        
        # Calculate RMSF
        rmsf = np.sqrt(np.mean(np.sum(fluctuations * fluctuations, axis=2)))
        
        return rmsf

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        headers.extend(["RMSD", "RMSF"])
        return headers

    @staticmethod
    def from_pdb(pdb: app.PDBFile, file: str, reportInterval: int, **kwargs):
        """Create an RMSDStateDataReporter from a PDB file.

        This method extracts the reference positions and protein atom indices
        from a PDB file. It identifies protein atoms by looking for standard
        protein residues (excluding water and ions).

        Args:
            pdb (app.PDBFile): PDB file
            file (str): The file to write the data to
            reportInterval (int): The interval (in time steps) at which to write frames
            **kwargs: Additional arguments to pass to StateDataReporter

        Returns:
            RMSDStateDataReporter: Configured reporter instance
        """
        # Get protein atom indices and positions
        protein_atoms = []

        # Standard protein residues (excluding water and ions)
        protein_residues = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
 
        # Get topology and positions
        topology = pdb.getTopology()
        positions = pdb.getPositions()

        # Iterate through residues to identify protein atoms
        for residue in topology.residues():
            if residue.name in protein_residues:
                for atom in residue.atoms():
                    protein_atoms.append(atom.index)

        # Get reference positions for protein atoms
        reference_positions = np.array(
            [positions[i].value_in_unit(unit.nanometers) for i in protein_atoms]
        )

        return RMSDStateDataReporter(
            file=file,
            reportInterval=reportInterval,
            reference_positions=reference_positions,
            protein_atoms=protein_atoms,
            **kwargs,
        )
