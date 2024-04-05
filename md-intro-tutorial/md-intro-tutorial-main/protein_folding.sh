# remove lines with HETATM and CONECT from the pdb file
!grep -v HETATM input/1fjs.pdb > 1fjs_protein_tmp.pdb
!grep -v CONECT 1fjs_protein_tmp.pdb > 1fjs_protein.pdb

    # create the three files using gmx pdb2gmx
    !gmx pdb2gmx -f 1fjs_protein.pdb -o 1fjs_processed.gro -water tip3p -ff "charmm27"

    # define the box: gmx editconf
    !gmx editconf -f 1fjs_processed.gro -o 1fjs_newbox.gro -c -d 1.0 -bt dodecahedron

    # fill the box with water: gmx solvate 
    !gmx solvate -cp 1fjs_newbox.gro -cs spc216.gro -o 1fjs_solv.gro -p topol.top

    # assembling / processing .tpr file with gmx grompp 
    !gmx grompp -f ions.mdp -c 1fjs_solv.gro -p topol.top -o ions.tpr

# create an empty mdp file so it can be used to create an ions.tpr file 
!touch ions.mdp

!printf "SOL\n" | gmx genion -s ions.tpr -o 1fjs_solv_ions.gro -conc 0.15 -p \
topol.top -pname NA -nname CL -neutral

    # prepare the .tpr binary input file
    !gmx grompp -f input/emin-charmm.mdp -c 1fjs_solv_ions.gro -p topol.top -o em.tpr

    # run mdrun to carry out the energy minimisation:
    !gmx mdrun -v -deffnm em

# extract potential energy to a .xvg file 
!printf "Potential\n0\n" | gmx energy -f em.edr -o potential.xvg -xvg none

    # MD Equilibrium run - temperature. 
    !gmx grompp -f input/nvt-charmm.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr 
    !gmx mdrun -ntmpi 1 -v -deffnm nvt

# extract temperature data
!echo "Temperature" | gmx energy -f nvt.edr -o temperature.xvg -xvg none -b 20

    # update the files using grompp
    !gmx grompp -f input/npt-charmm.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
    # start an MD simulation just like before
    !gmx mdrun -ntmpi 1 -v -deffnm npt

# extract pressure data
!echo "Pressure" | gmx energy -f npt.edr -o pressure.xvg -xvg none
# extract density data 
!echo "Density" | gmx energy -f npt.edr -o density.xvg -xvg none

    # the produciton run `grompp`
    !gmx grompp -f input/md-charmm.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
    # The production run `mdrun`
    !gmx mdrun -ntmpi 1 -v -deffnm md

# check Extract rmsd data
!printf "4\n1\n" | gmx rms -s em.tpr -f md_center.xtc -o rmsd_xray.xvg -tu ns -xvg none

# analyze the radius of gyration
!echo "1" | gmx gyrate -f md_center.xtc -s md.tpr -o gyrate.xvg -xvg none

# generate report 
!gmx report-methods -s md.tpr