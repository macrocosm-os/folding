# Mining Procedure
Unlike other subnets, miners are not required to run GPU-intensive jobs, but rather use high-performing CPUs to perform energy minimization routines for protein folding. Protein folding is a computationally intensive task. There are no shortcuts, and as such miners are rewarded truly on their ability to find the best configuration for their protein. 

Due to the fact we ask miners to run CPU-bound tasks, it allows validators to submit many jobs to miners with the anticipation that they will run these jobs in parallel *processes*. 

## Job Scheduling
The miner is responsible for running a `config.neuron.max_workers` number of pdb_id jobs, where these jobs are launched via the miner-specific logic. The flow is below: 

<div align="center">
    <img src="../assets/miner_flow.png" alt="Validator-flow">
</div>

Therefore, the more compute you have, the more pdb jobs you can support. The more jobs, the more opportunities you have for reward. 

The validator rewards miners by checking what each miner's intermediate results are for a specific pdb job. The faster you can find the best solution, the more likely you are to recieve rewards on each query step. 