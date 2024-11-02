The code here is written and tested in the following setting:
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe:/o:centos:centos:7
- Kernel: Linux 3.10.0-1127.19.1.el7.x86_64
- Architecture: x86-64
- Python version: 3.7.12

The codes are guranteed to work for the following package versions:
  - numpy=1.21.6 
  - pandas=1.3.5
  - torch==1.7.1

The input data consists of 2D Torch tensors, except for the chi statistic, which requires 1D integer tensors. For multinomial data with a large number of categories or continuous data with a dimensionality \(d\) and bin number \(\kappa\) such that \(\kappa^d\) is large, or when the sample size is very large,  for example when \(k = \kappa^d > 1000\) or \(n > 100,000\), we recommend using a GPU.

We recomment importing the conda environment from the following files:
- For Linux:
  - LDPUtsEnvK40.yaml
- For Windows:
  - LDPUtsEnvK40_windows.yaml


To replicate the simulation results in the paper, run the following Python files, adjusting the sample size, data dimension, and privacy parameters as specified in the files.
- For Figure 1:
  - paper_replication_type1.py 
- For Figure 2:
  - paper_replication_multinomial.py
- For Figure 3:
  - paper_replication_density_location.py
- For Figure 4:
  - paper_replication_appendix_g2.py
- For Figure 5:
  - paper_replication_density_scale.py

