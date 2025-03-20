# Fast Generation of Track and Vertex Quality Variables for Combinatorial Backgrounds
Bristol University Final Year Project - PPM12


## Overview


This project was initiated as part of efforts to improve simulation resources and acts as a sub-investigation into combinatorial backgrounds for fast track and vertex generation, pulling from Alex Marshall's fast_vtx, an editted fork of which is utilised in this directory:

`https://github.com/Viola03/fastVTX_fork`

### Objectives
- **Research:** To analyze current methodologies and apply them to a combinatorial context.
- **Development:** Creating proof-of-concept in a B decay chain case study.
- **Evaluation:** Simulate generation with RapidSim conditions and evaluate with BDTs.

## Project Notes and Rough Structure

```plaintext
ResearchProject/
├── datasets/            <-- full dataset, with splits, exploratory plots, validation
├── datasetsmixed/       <-- RapidSim mixed dataset, with exploratory plots, validation
├── inference/           <-- for running inference on defined dataset
│   ├── inference.py 
│   └── models/          <-- where .pkls are stored
├── model_test_runs/     <-- testing gpu_training
│   └── NewConditions_{num}/
│       └──READ          <-- explain architecture/training time
├── model_test_runs_expanded/  <-- more complete implementations 
├── model_final_runs/    <-- contains baseline model
│   └── hyperparameter_tuning
├── scripts/     
│   ├── validation/      <-- SignalBDT plots
│   ├── mixing/          <-- for proxy combinatorial RapidSim samples
│   └── various.py       <-- most tools (resampling B, renaming branches, etc.)
├── rapidsim/            <-- configs for rapidsim
├── *.yml         <-- dependencies
├── train_edit.py        <-- train model
├── save_networks.py     <-- save model
├── Alex_inference.py    <-- old inference script for reference
└── README.md
```

Note: path files were not all reviewed since gpu and often refer to local machines.
