from typing import List

import Bio
from Bio.PDB.vectors import rotaxis
from Bio.PDB import PDBParser, PDBIO, PPBuilder

import numpy as np


def rotate_atoms(atoms: List[Bio.PDB.Atom.Atom], rotation_axis, rotation_point, angle):
    """
    Rotate atoms around a rotation axis by a given angle.
    """
    rotation_matrix = rotaxis(angle, rotation_axis)
    for atom in atoms:
        coord = atom.get_vector() - rotation_point
        new_coord = coord.left_multiply(rotation_matrix) + rotation_point

        # This should be an in-place operation
        atom.set_coord(new_coord.get_array())


def modify_phi_angle(residue, pp, position, delta_phi):
    # Modify phi angle
    if residue.has_id("N") and residue.has_id("CA"):
        N = residue["N"].get_vector()
        CA = residue["CA"].get_vector()
        axis_phi = (CA - N).normalized()

        # Atoms to rotate for phi: all atoms after CA
        atoms_to_rotate_phi = []
        for res in pp[position:]:
            for atom in res.get_atoms():
                atoms_to_rotate_phi.append(atom)
        rotate_atoms(
            atoms=atoms_to_rotate_phi,
            rotation_axis=axis_phi,
            rotation_point=N,
            angle=delta_phi,
        )


def modify_psi_angle(residue, pp, position, delta_psi):
    # Modify psi angle
    if residue.has_id("CA") and residue.has_id("C"):
        CA = residue["CA"].get_vector()
        C = residue["C"].get_vector()
        axis_psi = (C - CA).normalized()

        # Atoms to rotate for psi: all atoms after C
        atoms_to_rotate_psi = []
        for res in pp[position + 1 :]:
            for atom in res.get_atoms():
                atoms_to_rotate_psi.append(atom)
        rotate_atoms(
            atoms=atoms_to_rotate_psi,
            rotation_axis=axis_psi,
            rotation_point=CA,
            angle=delta_psi,
        )


def unfold_protein(pdb_location: str, pbc: str = None):
    """Method to unfold the protein"""
    parser = PDBParser()
    structure = parser.get_structure(id="protein", file=pdb_location)
    ppb = PPBuilder()
    peptides = ppb.build_peptides(entity=structure, aa_only=1)
    # Iterate over all peptides
    for peptide in peptides:
        phi_psi = peptide.get_phi_psi_list()
        for i, residue in enumerate(peptide):
            phi, psi = phi_psi[i]
            # Skip residues where phi or psi is None
            if phi is None or psi is None:
                continue

            # Define new angles to simulate unfolding (e.g., set to 180 degrees)
            new_phi = np.deg2rad(180)
            new_psi = np.deg2rad(180)

            # Calculate angle differences
            delta_phi = new_phi - phi
            delta_psi = new_psi - psi

            # Modify phi angle
            modify_phi_angle(
                residue=residue, pp=peptide, position=i, delta_phi=delta_phi
            )

            # Modify psi angle
            modify_psi_angle(
                residue=residue, pp=peptide, position=i, delta_psi=delta_psi
            )

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_location)
        if pbc:
            with open(pdb_location, "r") as f:
                pdb = f.read()
            pdb_with_pbc = pbc + "\n" + pdb
            with open(pdb_location, "w") as f:
                f.write(pdb_with_pbc)
