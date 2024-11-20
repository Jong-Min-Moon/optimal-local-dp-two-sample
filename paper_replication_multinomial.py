from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import data_generator
import time
import numpy as np
from datetime import datetime

###################### Change settings #########################
param_dist    = 0.002        #perturbation magnitude from uniform distribution; choose from {0.04, 0.015, 0.002} 
d = 400                      # number of categories; choose from {4,40,400}
privacy_level = 1           # privacy level \alpha: choose from {0.5, 1, 2}
sample_size   = 1000        
n_permutation = 999         # fixed at 999 in the paper
priv_mech  = 'genrr' #choose among 'bitflip', 'genrr', 'lapu', 'disclapu'
statistic  = 'elltwo' #choose among 'chi', 'projchi', 'elltwo'. chi requires 1-dimensional multinomial data.
n_test        = 2000        
test_start    = 1
significance_level = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.get_num_threads())
###############################################################################

#################### create data generator, client, and server instances#############

data_gen = data_generator() #create data generator
LDPclient = client() #create the client, which privatizes the data

method_name = priv_mech + statistic

server_private_vec = {
    "elltwo":server_ell2(privacy_level),
    "chi":server_multinomial_genrr(privacy_level),
    "projchi":server_multinomial_bitflip(privacy_level)
    }
server_private = server_private_vec[statistic] #create the server, which conducts the test


#################### Run the simulations ##########################################
print(f"{method_name}, alpha={privacy_level}, sample size={sample_size}")
print("#########################################")
p_value_vec = np.zeros([n_test, 1])
statistic_vec = np.zeros([n_test, 1])
t = time.time()

for i in range(n_test):
    test_num = i + test_start
    t_start_i = time.time()
    torch.manual_seed(test_num)
    bump=param_dist
    p = torch.ones(d).div(d)
    p2 = p.add(
        torch.remainder(
        torch.tensor(range(d)),
        2
        ).add(-1/2).mul(2).mul(bump)
    )
    p1_idx = torch.cat( ( torch.arange(1, d), torch.tensor([0])), 0)
    p1 = p2[p1_idx]


    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p1, sample_size),
            d,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p2, sample_size),
            d,
            privacy_level,
            device
        ),
    d,
    device,
    device
    )

    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    p_value_vec[i], statistic_vec[i] = server_private.release_p_value_permutation(n_permutation)
    t_end_i = time.time() - t_start_i

    print(f"pval: {p_value_vec[i]} -- {test_num}th test, time elapsed {t_end_i} -- emperical power so far (from test_start): {(p_value_vec[0:(i+1)] < significance_level).mean()}")

