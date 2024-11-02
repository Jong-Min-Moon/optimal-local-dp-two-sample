from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import data_generator
import time
import numpy as np
from datetime import datetime

###################### Change settings #########################
d = 3                       # data dimension; choose from {3,4,5}
n_bin = 4                   # fixed at 4 in the paper
privacy_level = 1           # privacy level \alpha: choose from {0.5, 1, 2}
sample_size   = 1000        
n_permutation = 999         # fixed at 999 in the paper
priv_mech  = 'genrr' #choose among 'bitflip', 'genrr', 'lapu', 'disclapu'
statistic  = 'elltwo' #choose among 'chi', 'projchi', 'elltwo'. chi requires 1-dimensional multinomial data.
n_test        = 2000        
test_start    = 1
significance_level = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############################################################################

print(torch.get_num_threads())
data_gen = data_generator()
LDPclient = client()
method_name = priv_mech + statistic

server_private_vec = {
    "elltwo":server_ell2(privacy_level),
    "chi":server_multinomial_genrr(privacy_level),
    "projchi":server_multinomial_bitflip(privacy_level)
    }
server_private = server_private_vec[statistic]

print(f"{method_name}, alpha={privacy_level}, sample size={sample_size}")
print("#########################################")
p_value_vec = np.zeros([n_test, 1])
statistic_vec = np.zeros([n_test, 1])
t = time.time()

for i in range(n_test):
    test_num = i + test_start
    t_start_i = time.time()
    torch.manual_seed(test_num)
    copula_mean_1 = -0.5 * torch.ones(d).to(device)
    copula_mean_2 =  -copula_mean_1


    copula_sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)
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
    server_private.load_private_data_multinomial(
        data_y_priv,
        data_z_priv,
        LDPclient.alphabet_size_binned,
        device,
        device
    )
    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    p_value_vec[i], statistic_vec[i] = server_private.release_p_value_permutation(n_permutation)
    t_end_i = time.time() - t_start_i
    print(f"pval: {p_value_vec[i]} -- {test_num}th test, time elapsed {t_end_i} -- emperical power so far (from test_start): {(p_value_vec[0:(i+1)] < significance_level).mean()}")
