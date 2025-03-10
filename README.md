# Fast Generation of Track and Vertex Quality Variables for Combinatorial Backgrounds
Bristol University Final Year Project - PPM12


## Overview


This project was initiated as part of ongoing efforts to improve simulation resources and acts as a sub-investigation into combinatorial backgrounds for fast track and vertex generation, pulling from Alex Marshall's fast_vtx, an editted fork of which is utilised in this directory:

`https://github.com/Viola03/fastVTX_fork`

### Objectives
- **Research:** To analyze current methodologies and propose new approaches.
- **Development:** To create prototypes and proof-of-concept implementations.
- **Evaluation:** To rigorously test and evaluate the outcomes against standard benchmarks.

## Project Notes and Rough Structure

```plaintext
ResearchProject/
├── datasets/
├── datasetsmixed/
├── scripts/              
│   ├── mixing/          <-- for proxy combinatorial RapidSim samples
│   └── various.py       <-- most tools (resampling B, renaming branches, etc.)
├── rapidsim/            <-- configs for rapidsim
├── test_runs_branches/  <-- testing gpu_training
│   ├── NewConditions_{num}/
│       └──READ          <-- explain architecture/training time
├── test_runs_expanded/  <-- more complete implementations 
│   ├── test_module1.py
│   └── test_module2.py
├── final_run/
├── inference/
│   ├── models/          <-- where .pkls are stored
├── fast_vtx.yml         <-- dependencies
├── train_edit.py        <-- train model
├── save_networks.py     <-- save model
├── run_inference.py     <-- run inference on defined dataset
└── README.md
```

Note: path files were not all reviewed and often refer to local machines.
