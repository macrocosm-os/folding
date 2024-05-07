<!-- <div align="center">
    <img src="./assets/macrocosmos-black.png" alt="Alt generative-folding-tao">
</div> -->

<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <img src="macrocosmos-white.png">
</picture>

<picture>
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Protein Folding Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

This repository is the official codebase for Bittensor Subnet Folding (SNX), which is to be released May 2024. To learn more about the Bittensor project and the underlying mechanics, [read here.](https://docs.bittensor.com/)

---

<div align="center">
    <img src="./assets/protein_tao.png" alt="Alt generative-folding-tao">
</div>

# Introduction
The protein folding subnet is Bittensors’ first venture into academic use cases, built and maintained by [Macrocosmos AI](https://www.macrocosmos.ai). While the current subnet landscape consists of mainly AI and web-scraping protocols, we believe that it is important to highlight to the world how Bittensor is flexible enough to solve almost any problem.

This subnet is designed to produce valuable academic research in Bittensor. Researchers and universities can use this subnet to solve almost any protein, on demand, for free. It is our hope that this subnet will empower researchers to conduct world-class research and publish in top journals while demonstrating that decentralized systems are an economic and efficient alternative to traditional approaches.

  
# What is Protein Folding?  
  
  Proteins are the biological molecules that "do" things, they are the molecular machines of biochemistry. Enzymes that break down food, hemoglobin that carries oxygen in blood, and actin filaments that make muscles contract are all proteins. They are made from long chains of amino acids, and the sequence of these chains is the information that is stored in DNA. However, its a large step to go from a 2D chain of amino acids to a 3D structure capable of working. 

  The process of this 2D structure folding on itself into a stable, 3D shape is called **protein folding**. For the most part, this process happens naturally and the end structure is in a much lower free energy state than the string. Like a bag of legos though, it is not enough to just know the building blocks being used, its the way they're supposed to be put together that matters. *"Form defines function"* is a common phrase in biochemsitry, and it is the quest to determine form, and thus function of proteins, that makes this process so important to understand and simulate. 

# Why is Folding a Good Subnet Idea? 
An ideal incentive mechanism defines an asymmetric workload between the validators and miners. The necessary proof of work (PoW) for the miners must require substantial effort and should be impossible to circumvent. On the other hand, the validation and rewarding process should benefit from some kind of privileged position or vantage point so that an objective score can be assigned without excess work. Put simply, **rewarding should be objective and adversarially robust**.

Protein folding is also a research topic that is of incredibly high value. Research groups all over the world dedicate their time to solving particular niches within this space. Providing a solution to attack this problem at scale is what Bittensor is meant to provide to the global community. 

# Reward Mechanism
Protein folding is a textbook example of this kind of asymmetry; the molecular dynamics simulation involves long and arduous calculations which apply the laws of physics to the system over and over again until an optimized configuration is obtained. There are no reasonable shortcuts. 

While the process of simulation is exceedingly compute-intensive, the evaluation process is actually straightforward. **The reward given to the miners is based on the ‘energy’ of their protein configuration (or shape)**. The energy value compactly represents the overall quality of their result, and this value is precisely what is decreased over the course of a molecular dynamics simulation. The energy directly corresponds to the configuration of the structure, and can be computed in closed-form. The gif below illustrates the energy minimization over a short simulation procedure.

<div align="center">
    <img src="./assets/8emf_pdb_loss.gif" alt="Alt Folded-protein" width="500" height="350">
</div>

When the simulations finally converge (ΔE/t < threshold), they produce the form of the proteins as they are observed in real physical contexts, and this form gives rise to their biological function. Thus, the miners provide utility by preparing ready-for-study proteins on demand. An example of such a protein is shown below. 

<div align="center">
    <img src="./assets/8emf_pdb_protein.gif" alt="Alt Folded-protein" width="600" height="500">
</div>

# Running the Subnet
## Requirements 
Protein folding utilizes a standardized package called [GROMACS](https://manual.gromacs.org/2023.2/install-guide/index.htm). To run, you will need:
1. A Linux-based machine 
2. Multiple CPU cores. 

Out of the box, we do not require miners to run GPU compatible GROMACS packages.

## Installation
This repository requires python3.8 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/macrocosm-os/folding.git
cd folding
<OPTIONAL CREATE VIRTUAL ENVIRONMENT (RECOMMENDED)>
pip install -r requirements.txt
bash install.sh
pip install -e .
```

## How does the Subnet Work?

In this subnet, validators create protein folding challenges for miners, who in turn run simulations based on GROMACS to obtain stable protein configurations. 

### Validation

Each round of validation consists of the following steps:
1. Validator randomly select a protein ID (aka. `pdb_id`) from a large database of options.
2. Validator downloads the `.pdb` file for the selected `pdb_id`, which contains the initial coordinates of the protein.
3. Validator runs some validation and preprocessing scripts to ensure that the problem is well posed. This includes a brute-force search over a specified space of hyperparameters to effectively search over a wide range of simulation configurations. 
4. The validator choses an N number of miners on the network to complete the folding job for this particular pdb_id. Once submitted, the validator keeps track of which miners are running a submitted pdb_id, and continues to submit jobs to the network until miners are at full capacity. 
5. The validator periodically checks up on the miners for **each** pdb_id job it has submitted, evaluating the state of each miners intermediate/final response during the query.
6. Validators confirm that miners are running the correct protein through some GROMACS commands, and miners are graded by their protein configuration (energy). The miner with the lowest energy on each step will be rewarded. 


## Mining
Each round of mining consists of the following steps:
1. Miner receives the input files to run a simulation for a `pdb_id`.
2. Miner runs **first** (_low_ fidelity) `mdrun` simulation using GROMACS which we call the temperature run.
3. Miner runs **second** (_medium_ fidelity) `mdrun` simulation based on temperature run results, which we call the temperature + pressure run.
4. Miner runs **third** (_high_ fidelity) `mdrun` simulation based on temperature + pressure run results, which we call the production run. This run lasts the longest.
5. Miners are expected to be queried by the validator at a frequency unknown to them. They are expected to respond with the files necessary for the validator to confirm that they are running the correct protein and for their energy-based calculations. 
6. Importantly, miners run an N number of **processes**. Therefore, miners are expected to handle running multiple pdb simulations concurrently. 

## Notes

**Miner** simulations will output a projected time. The first two runs will be about the same length, with the third taking about an order of magnitude longer using a default number of steps = 50,000. The number of steps (`steps`) and the maximum allowed runtime (`maxh`) are easily configurable and should be employed by miners to prevent timing out. We also encourage miners to take advantage of 'early stopping' techniques so that simulations do not run past convergence.

Furthermore, we want to support the use of ML-based mining so that recent algorithmic advances (e.g. AlphaFold) can be leveraged. At present, this subnet is effectively a **specialized compute subnet** (rather than an algorithmic subnet). For now, we leave this work to motivated miners.

GROMACS itself is a rather robust package and is widely used within the research community. There are specific guides and functions if you wish to parallelize your processing or run these computations off of a GPU to speed things up.



## License

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
