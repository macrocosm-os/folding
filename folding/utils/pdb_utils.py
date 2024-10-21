from typing import List

import Bio
from Bio.PDB.vectors import rotaxis


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
