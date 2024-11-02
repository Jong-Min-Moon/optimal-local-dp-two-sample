The code is written and tested in the following environment:

- **Operating System**: CentOS Linux 7 (Core)  
- **CPE OS Name**: `cpe:/o:centos:centos:7`  
- **Kernel**: `Linux 3.10.0-1127.19.1.el7.x86_64`  
- **Architecture**: `x86-64`  
- **Python Version**: 3.7.12  

The code is guaranteed to work with the following package versions:

- `numpy==1.21.6`
- `pandas==1.3.5`
- `torch==1.7.1`

### Data Requirements

The input data consists of 2D Torch tensors, except for the chi statistic, which requires 1D integer tensors. For multinomial data with a large number of categories, or for continuous data with high dimensionality (`d`) and bin number (`κ`) such that `κ^d` is large, or when the sample size is very large (e.g., `k = κ^d > 1000` or `n > 100,000`), we recommend using a GPU.

### Conda Environment Setup

We recommend importing the conda environment from the following files:

- **For Linux**: `LDPUtsEnvK40.yaml`
- **For Windows**: `LDPUtsEnvK40_windows.yaml`

### Reproducing Simulation Results

To replicate the simulation results in the paper, run the following Python files. Adjust the sample size, data dimension, and privacy parameters as specified in each file:

- **Figure 1**: `paper_replication_type1.py`
- **Figure 2**: `paper_replication_multinomial.py`
- **Figure 3**: `paper_replication_density_location.py`
- **Figure 4**: `paper_replication_appendix_g2.py`
- **Figure 5**: `paper_replication_density_scale.py`