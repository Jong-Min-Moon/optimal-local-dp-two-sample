### Import packages
from client import client
import torch
from server import server_ell2, server_multinomial_genrr, server_multinomial_bitflip
from data_generator import data_generator
import time
import numpy as np


### Change settings ################
sample_size = 500 # choose from {500, 1000, 1500}
n_test = 10
n_permutation = 999
significance_level = 0.05 # change continuously from .01 to .99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Fixed settings
d = 500 #fixed at k=500 in the paper
privacy_level = 0.1 # fixed at alpha=0.1 in the paper


### choose between the following two settings of the probability vector:
p = torch.arange(1,d).float().reciprocal() #multinomial probability vector for the power law setting
#p = torch.ones(d) # multinomial probability vector for the uniform law setting
#####################################################
p = p.divide(p.sum())


### create data generator, client, and server instances
data_gen = data_generator() #create data generator
LDPclient = client() #create the client, which privatizes the data
server_private_vec = {
    "elltwo":server_ell2(privacy_level),
    "chi":server_multinomial_genrr(privacy_level),
    "projchi":server_multinomial_bitflip(privacy_level)
    }
server_list = ["elltwo", "chi", "projchi"]
privmech_list = ["lapu", "genrr", "bitflip"]
p_value_array = np.zeros([n_test, 3])


### Lapu + elltwo, permutation test

t = time.time()
for i in range(n_test):
    print(f"{i+1}th test")
    torch.manual_seed(i)
    server_private = server_private_vec["elltwo"] 
    privmech_now = "lapu"
    
    
    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        d,
        device,
        device
        )
    p_value_array[i,0], _ = server_private.release_p_value_permutation(n_permutation)

elapsed = time.time() - t
print(elapsed)


### genrr + chi, chi-square asymptotics
t = time.time()
for i in range(n_test):
    print(f"{i+1}th test")
    torch.manual_seed(i)
    server_private = server_private_vec["chi"] 
    privmech_now = "genrr"
    
    
    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        d,
        device,
        device
        )
    p_value_array[i,1], _ = server_private.release_p_value()

elapsed = time.time() - t
print(elapsed)


### rappor + projchi, chi-square asymptotics
t = time.time()
for i in range(n_test):
    print(f"{i+1}th test")
    torch.manual_seed(i)
    server_private = server_private_vec["projchi"] 
    privmech_now = "bitflip"
    
    
    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            privmech_now,
            data_gen.generate_multinomial_data(p, sample_size),
            d,
            privacy_level,
            device
        ),
        d,
        device,
        device
        )
    p_value_array[i,2], _ = server_private.release_p_value()

elapsed = time.time() - t
print(elapsed)