<div align="center">

# **Protien Folding Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
- [Background: What is protein folding?](#background)
- [Description](#description)
- [Notes](#notes)

- [License](#license)



## Introduction

  This is the start of the protein folding subnet. This subnet uses the GROMACS software to simulate molecular dynamics of proteins. We take a known initial 3D structure, and put in a cell-like environment and simluate it to know its end form. This is an essential step in the protein folding process and an entry point to many other high level techniques.

This subnet is designed to produce valuable academic research, but has been designed to be accessible to anyone as mining and validating does not require any background knowledge of molecular dynamics simulations.

General information about GROMACS can be found here: https://manual.gromacs.org/2023.2/index.html

__Before you proceed__

  Complexity of the problem aside, one of the current barriers to protein folding is **computational ability**. The processes involved are complex and take time even with state of the art systems. In contrast to other subnets, a single mining step can take hours. 
  
  
## Background  
  
  Proteins are the biological molecules that "do" things, they are the molecular machines of biochemistry. Enzymes that break down food, hemoglobin that carries oxygen in blood, and actin filaments that make muscles contract are all proteins. They are made from long chains of amino acids, and the sequence of these chains is the information that is stored in DNA. However, its a large step to go from a 2D chain of amino acids to a 3D structure capable of working. 

  The process of this 2D structure folding on itself into a stable, 3D shape in a cell is called protein folding. For the most part, this process happens naturally and the end structure is in a much lower free energy state than the string. Like a bag of legos though, its not enough to just know the building blocks being used, its the way they're supposed to be put together that matters. "Form defines function" is a common phrase in biochemsitry, and it is the quest to determine form, and thus function of proetins, that makes this process so important to understand and simulate. 

  Understanding how specific proteins fold unlocks the ability to cure many ailments. Folding@Home, a distributed computing community dedicated to simulating protein folding, was able to help design a treatment for SARS-covid-19 by identifying a unique folding pattern in the spike protein of the virus that left it open to interference. Understanding how beta amyloid plaques fold, and thus misfold, is essential to understanding how Alzheimers Disease develops and to identify potential treamtent protocols.


## Installation
### GROMACS
You will need two packages to run either a miner or a validator. GROMACS itself, and then a GROMACS wrapper to make the base functions more python friendly. You can find the install process and requirements for the latest version of GROMACS here:
- `https://manual.gromacs.org/2023.2/install-guide/index.html`

However, I found package managers make the process much simpler based on your preffered workflow:
- Conda install: `conda install -c conda-forge gromacs`
- Brew install: `brew install gromacs`

### GromacsWrapper
For the most part, this leaves the base syntax intact. Installation instructions and more can be found here: https://gromacswrapper.readthedocs.io/en/latest/installation.html
- Conda install: `conda install -c conda-forge gromacswrapper`
- pip install: `pip install GromacsWrapper`



## Description

In this subnet, validators create protein folding challenges for miners, who in turn run simulations based on GROMACS to obtain stable protein configurations. 

### Validation

Each round of validation consists of the following steps:
1. Validator randomly select a protein ID (aka. `pdb_id`) from a large database of options.
2. Validator downloads the `.pdb` file for the selected `pdb_id`, which contains the initial coordinates of the protein.
3. Validator runs some validation and preprocessing scripts to ensure that the problem is well posed.
4. Validator sends input files to miners and waits until `timeout` is reached before optimized coordinates are returned.
5. Validator scores miner responses based on optimality of protein coordinates.

__Reward__:
After verifying that miners performed the required computation, the free energy of the protein is calculated based on the output file. The free energy will converge to a minimum value when the optimal protein configuration is obtained, and so each miner's rank is based on the optimality of their coordinates.

## Mining
Each round of mining consists of the following steps:
1. Miner receives the input files for a `pdb_id`.
2. Miner runs **first** (_low_ fidelity) `mdrun` simulation using GROMACS which we call the temperature run.
3. Miner runs **second** (_medium_ fidelity) `mdrun` simulation based on temperature run results, which we call the temperature + pressure run.
4. Miner runs **third** (_high_ fidelity) `mdrun` simulation based on temperature + pressure run results, which we call the production run. This run lasts the longest.
5. Miner responds with optimized protein coordinates from last run (or earlier run if not available).

## Notes
Several key inputs such as the `forcefield` and simulation `box` shape are kept fixed in the present version, but will likely become additional variables in a future version.
- Forcefield: "Charmm27"
- Box: "Rhombic dodecahedron"


**Miner** simulations will output a projected time. The first two runs will be about the same length, with the third taking about an order of magnitude longer using a default number of steps = 50,000. The number of steps (`steps`) and the maximum allowed runtime (`maxh`) are easily configurable and should be employed by miners to prevent timing out. We also encourage miners to take advantage of 'early stopping' techniques so that simulations do not run past convergence.

Furthermore, we want to support the use of ML-based mining so that recent algorithmic advances (e.g. AlphaFold) can be leveraged. At present, this subnet is effectively a **specialized compute subnet** (rather than an algorithmic subnet). For now, we leave this work to motivated miners.

GROMACS itself is a rather robust package and is widely used within the research community. There are specific guides and functions if you wish to parallelize your processing or run these computations off of a GPU to speed things up.



## License

[INSERT GROMACS LICENSING INFORMATION HERE]



This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
