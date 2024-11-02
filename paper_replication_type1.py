import sys

from discretizer import discretizer
from client import client
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from server import server_ell2, server_multinomial_genrr, server_multinomial_bitflip
from data_generator import data_generator
import time
import numpy as np
from scipy.stats import chi2
from utils import chi_sq_dist
alphabet_size = 500 #fixed at k=500 in the paper

################### Change settings ################
sample_size = 500 # choose from {500, 1000, 1500}
# choose between the following two lines:
p = torch.arange(1,alphabet_size).reciprocal() #multinomial probability vector for the power law setting
#p = torch.ones(alphabet_size) # multinomial probability vector for the uniform law setting
#####################################################
p = p.divide(p.sum())
n_test = 2000
n_permutation = 999
significance_level = 0.05

privacy_level = 0.1 # fixed at alpha=0.1 in the paper


data_gen = data_generator(device)




p_value_array = np.zeros([n_test, 3])
LDPclient = client(device, privacy_level)
server_elltwo = server_ell2(device, privacy_level)
server_genrr = server_multinomial_genrr(device, privacy_level)
server_bitflip = server_multinomial_bitflip(device, privacy_level)

t = time.time()
for i in range(n_test):
    print(f"{i+1}th test")
    torch.manual_seed(i)
    data_y = data_gen.generate_multinomial_data(p, sample_size)
    data_z = data_gen.generate_multinomial_data(p, sample_size)
    
    LDPclient.load_data_multinomial(data_y, data_z, alphabet_size)
    data_list_lapu_y, data_list_lapu_z = LDPclient.release_lapu()
    data_list_genrr_y, data_list_genrr_z = LDPclient.release_genrr()
    data_list_bitflip_y, data_list_bitflip_z = LDPclient.release_bitflip()
    
    server_elltwo.load_private_data_multinomial(data_list_lapu_y, data_list_lapu_z, alphabet_size)
    p_value_array[i,0] = server_elltwo.release_p_value_permutation(n_permutation)
 
    server_genrr.load_private_data_multinomial(data_list_genrr_y, data_list_genrr_z, alphabet_size)
    p_value_array[i,1] = server_genrr.release_p_value()
     
    server_bitflip.load_private_data_multinomial(data_list_bitflip_y, data_list_bitflip_z, alphabet_size)
    p_value_array[i,2] = server_bitflip.release_p_value()
elapsed = time.time() - t
print(elapsed)
print(
   # f"small chi-square distance\n"+
        f"privacy level = {privacy_level}"
)


