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

The input data are 2d torch tensors, except for chi statistic which takes 1d integer tensors.

We recomment importing the conda environment from the following files:
\n
For Linux:
 - LDPUtsEnvK40.yaml
\n
For Windows:
 - LDPUtsEnvK40_windows.yaml


 You can replicate the simulation results in the paper by running the following python files, with changed sample size, data dimension and privacy parameters (options are provided in the files)
 \n
 For Figure 1:
 - paper_replication_type1.py 
 \n
 For Figure 2:
 - paper_replication_multinomial.py
 \n
 For Figure 3:
 - paper_replication_density_location.py
 \n
 For Figure 4:
  - paper_replication_appendix_g2.py
  \n
 For Figure 5:
 - paper_replication_density_scale.py

