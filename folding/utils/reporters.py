import os
import openmm.app as app
import numpy as np
from openmm import unit
import pandas as pd


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


class RMSDReporter(object):
    """A reporter that saves RMSD measurements against a reference structure.
    
    This reporter calculates and saves RMSD values between current protein positions
    and a reference structure, excluding solvent and ions.
    """
    
    def __init__(self, file, reportInterval, reference_positions, protein_atoms):
        """Initialize the reporter.

        Args:
            file (str): The file to write the RMSD data to
            reportInterval (int): The interval (in time steps) at which to write frames
            reference_positions (np.ndarray): Reference positions to calculate RMSD against
            protein_atoms (list): List of atom indices that belong to the protein
        """
        self.file = file
        self.reportInterval = reportInterval
        self.reference_positions = reference_positions
        self.protein_atoms = protein_atoms
        self.rmsds = pd.DataFrame(columns=['step', 'rmsd'])

    @staticmethod
    def from_pdb(pdb: app.PDBFile, file: str, reportInterval: int):
        """Create an RMSDReporter from a PDB file.

        This method extracts the reference positions and protein atom indices
        from a PDB file. It identifies protein atoms by looking for standard
        protein residues (excluding water and ions).

        Args:
            pdb (app.PDBFile): PDB file
            file (str): The file to write the RMSD data to
            reportInterval (int): The interval (in time steps) at which to write frames

        Returns:
            RMSDReporter: Configured reporter instance
        """
        # Get protein atom indices and positions
        protein_atoms = []

        # Standard protein residues (excluding water and ions)
        protein_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
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
            [positions[i].value_in_unit(unit.nanometer) for i in protein_atoms]
        )

        return RMSDReporter(
            file=file,
            reportInterval=reportInterval,
            reference_positions=reference_positions,
            protein_atoms=protein_atoms
        )

    def describeNextReport(self, simulation):
        """Get information about the next report that will be generated.

        Args:
            simulation (Simulation): The Simulation to generate a report for

        Returns:
            tuple: A 5 element tuple containing (steps, positions, velocities, forces, energies)
        """
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        """Generate a report.

        Args:
            simulation (Simulation): The Simulation to generate a report for
            state (State): The current state of the simulation
        """
        # Get current positions
        positions = simulation.context.getState(getPositions=True).getPositions()

        # Extract protein positions
        protein_positions = np.array(
            [positions[i].value_in_unit(unit.nanometer) for i in self.protein_atoms]
        )

        # Calculate RMSD
        rmsd = self._calculate_rmsd(protein_positions)
        new_row = pd.DataFrame({'step': [simulation.currentStep], 'rmsd': [rmsd]})
        self.rmsds = pd.concat([self.rmsds, new_row], ignore_index=True)
        self.rmsds.to_csv(self.file, index=False)
        
    def _calculate_rmsd(self, positions):
        """Calculate RMSD between current and reference positions.

        Args:
            positions (np.ndarray): Current positions

        Returns:
            float: RMSD value in nanometers
        """
        # Center the structures
        ref_centered = self.reference_positions - np.mean(self.reference_positions, axis=0)
        pos_centered = positions - np.mean(positions, axis=0)

        # Calculate RMSD
        diff = ref_centered - pos_centered
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

        return rmsd

    def finalize(self):
        """Clean up by closing the output file."""
        if hasattr(self, '_outstream'):
            self._outstream.close()
