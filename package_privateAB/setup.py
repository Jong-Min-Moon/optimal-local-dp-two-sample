from setuptools import setup, find_packages

setup(
    name='privateAB',
    version='0.0.2',
    description='Two-sample testing (A/B testing) for multinomial and multivariate continuous data under local differential privacy',
    author='Jongmin Mun',
    author_email='jongmin.mun@marshall.usc.edu',
    url='https://jong-min-moon.github.io/softwares/',
    project_urls={
        "Bug Tracker": "https://github.com/Jong-Min-Moon/optimal-local-dp-two-sample",
    },
    install_requires = ['torch>=1.7.1', 'scipy>=1.7.3', 'numpy>=1.21.6', 'pandas>=1.3.5'],
    packages=find_packages(exclude=[]),
    keywords=['local differential privacy', 'A/B test', 'two-sample test', 'permutation test'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description="""
# package `privateAB`:  two-sample testing under local differential privacy
The package `privateAB` and the codes in this repository implement the private testing method introduced in the paper *Minimax Optimal Two-Sample Testing under Local Differential Privacy*, authored by Jongmin Mun, Seungwoo Kwak, and Ilmun Kim.  

The full paper can be accessed at: [https://arxiv.org/abs/2411.09064](https://arxiv.org/abs/2411.09064).  



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


## Basic usage
Two main objects are utilized in this package: `client`, which implements the privacy mechanism, and `server`, which conducts the test.
### Installation
```
pip install privateAB
```
###  Privatization of multinomial data
`client` takes raw data in the form of a PyTorch tensor and releases its locally differentially private representation.  

In this example, we use the `data_generator` function from our paper, which internally utilizes the `torch.multinomial` function. Therefore, when using your own data, ensure it follows the same format as the output of `torch.multinomial`.  

To get started, first import the necessary packages:
```
from privateAB.client import client
from privateAB.data_generator import data_generator
```
Now, using our `data_generator` function, we generate two independent datasets of multinomial samples.
```
import torch
#set probability vectors
sample_size   = 1000
d = 4 #number of categories of the multinomial data
param_dist    = 0.04 
p = torch.ones(d).div(d)
p2 = p.add(
        torch.remainder(
        torch.tensor(range(d)),
        2
        ).add(-1/2).mul(2).mul(bump)
    )
p1_idx = torch.cat( ( torch.arange(1, d), torch.tensor([0])), 0)
p1 = p2[p1_idx]

#create the data_generator instance
data_gen = data_generator() 

# generate raw data 
raw_data_1 = data_gen.generate_multinomial_data(p1, sample_size)
raw_data_2 = data_gen.generate_multinomial_data(p2, sample_size)
```
Next, we create an instance of the `client` class and use its `release_private` method to privatize the raw data.  

The `release_private` method requires the following five inputs:  
1. **Privacy mechanism**: A string specifying the mechanism to use ('bitflip', 'genrr', 'lapu', or 'disclapu').  
2. **Raw data**: A `torch.tensor` object representing the input data.  
3. **Number of categories**: The number of categories in the multinomial data.  
4. **Privacy parameter**: The parameter controlling the level of local differential privacy.  
5. **Device**: The computational device to be used ('cpu' or 'gpu') as supported by `torch`.
```
LDPclient = client() #create the client, which privatizes the data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #specify gpu or cpu

priv_mech  = 'bitflip' #choose among 'bitflip', 'genrr', 'lapu', 'disclapu'. bitflip corresponds to rappor in the paper.

private_data_1 = LDPclient.release_private(
            priv_mech,
            raw_data_1,
            d,
            0.9,
            device
        )
private_data_2 = LDPclient.release_private(
            priv_mech,
            raw_data_2,
            d,
            0.9,
            device
        )
```
###  Testing of multinomial data
The test is conducted using one of the following server instances: `server_multinomial_bitflip`, `server_ell2`, or `server_multinomial_genrr`. These correspond to the ProjChi, ell2, and Chi statistics discussed in the paper.  

- The first two servers (`server_multinomial_bitflip` and `server_ell2`) can process privatized views generated using the 'bitflip', 'lapu', or 'disclapu' mechanisms.  
- The `server_multinomial_genrr` instance, however, exclusively supports privatized views generated by the 'genrr' mechanism.  

To proceed, we first create a server instance, which requires the privacy parameter as input. Next, we load the privatized data using the `load_private_data_multinomial` method. This method takes the following five inputs:  
1. **First private data object**: The first dataset's privatized representation.  
2. **Second private data object**: The second dataset's privatized representation (for A/B testing).  
3. **Number of categories**: The number of categories in the multinomial data.  
4. **Device for the first private data**: The `torch` device (CPU or GPU) used to process the first dataset.  
5. **Device for the second private data**: The `torch` device used to process the second dataset.  

We allow two separate devices to accommodate large-scale datasets where GPU memory might be limited, requiring the calculations to be performed separately for each of the two data set. However, you can use the same device for both datasets if memory is not a concern.
```
from privateAB.server import server_multinomial_bitflip
server_multinomial_bitflip(0.9) #create an instance
server_private.load_private_data_multinomial(
    private_data_1, private_data_2 ,
    d,
    device,
    device
    )
```
Now we run the test. Any of the server instances (`server_ell2`, `server_multinomial_bitflip`, or `server_multinomial_genrr`) can calculate the permutation p-value using the `release_p_value_permutation` method.  

This method takes a single input:  
- **Number of permutations**: The number of permutations to perform.  

It returns two outputs:  
1. **p-value**: The significance level of the test.  
2. **Test statistic value**: The calculated value of the test statistic.
```
p_value, statistic = server_private.release_p_value_permutation(n_permutation)
```
`server_multinomial_bitflip` and `server_multinomial_genrr` can also compute the p-value based on the asymptotic chi-square null distribution using the `release_p_value` method.  

This method does not require any input arguments. It directly outputs:  
1. **p-value**: The significance level based on the chi-square null distribution.
```
p_value, statistic = server_private.release_p_value()
```

###  Privatization of continuous data
As discussed in our paper, the privatization of continuous data uses a binning method. We support data in the form of a $d$-dimensional PyTorch tensor, where each dimension falls within the interval $[0,1]$. If your data lies outside this range, you should apply an appropriate transformation, such as the CDF transformation mentioned in our paper.  

For convenience, we use our `data_generator` function to create two sets of multivariate continuous data. This function ensures the generated data adheres to the required format and simplifies the process of preparing data for privatization.  
```
import torch
from privateAB.data_generator import data_generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #specify gpu or cpu

d=3
copula_mean_1 = -0.5 * torch.ones(d).to(device)
copula_mean_2 =  -copula_mean_1
copula_sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)
data_gen = data_generator()
raw_data_1 = data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, copula_sigma)
raw_data_2 = data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, copula_sigma)
```
Now we privatize the multivariate continuous data using the `release_private_conti` method. This method is similar to `release_private` but automatically detects the data's dimensionality. Instead of specifying the number of categories, you provide the number of bins for discretizing the data.  

The `release_private_conti` method requires the following five inputs:  
1. **Privacy mechanism**: A string specifying the mechanism to use ('bitflip', 'genrr', 'lapu', or 'disclapu').  
2. **Raw data**: A `torch.tensor` object representing the input multivariate continuous data.  
3. **Number of bins**: The number of bins to discretize each dimension of the data.  
4. **Privacy parameter**: The parameter controlling the level of local differential privacy.  
5. **Device**: The computational device to be used ('cpu' or 'gpu') as supported by `torch`.  
```
privacy_level=0.9
n_bin=4
data_y_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, copula_sigma),
            privacy_level,
            n_bin,
            device
        )

data_z_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, copula_sigma),
            privacy_level,
            n_bin,
            device
        )

```
###  Testing of continuous data
After privatization, the data format aligns with that of multinomial data, allowing the same testing procedures to be applied.  

One important note is that the **number of categories** should equal the bin number raised to the power of the data dimension. You don’t need to calculate this manually, as it is automatically stored in `LDPclient.alphabet_size_binned`. This ensures consistency and simplifies the setup for testing.  




    


## Reproducing Simulation Results

To replicate the simulation results in the paper, run the following Python files. Adjust the sample size, data dimension, and privacy parameters as specified in each file:

- **Figure 2**: `Figure2_type_I.py` or `Figure2_type_I.ipynb` 
- **Figure 3**: `Figure3_multinomial.py` or `Figure3_multinomial.ipynb`
- **Figure 4**: `Figure4_density_location.py` or `Figure4_density_location.ipynb`
- **Figure 5**: `Figure5_rappor_elltwo_vs_projchi.py` or `Figure5_rappor_elltwo_vs_projchi.ipynb`
- **Figure 6**: `Figure6_density_scale.py` or `Figure6_density_scale.ipynb`
"""    
,
    long_description_content_type='text/markdown'
    # ....
)
