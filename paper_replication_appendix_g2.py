from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import create_power_law, data_generator
import time
import numpy as np
from datetime import datetime

###################### Change settings #########################
 
param_dist        = 2.45 # power law parameter of distribution Y; fixed at 2.45 in the paper
power_2           = 2.3  # power law parameter of distribution Z; fixed at 2.3 in the paper
d = 40                      # number of categories; fixed at 40 in the paper
privacy_level = 1           # privacy level \alpha: choose from {1, 2, 4}
sample_size   = 8400        
n_permutation = 999         # fixed at 999 in the paper
priv_mech  = 'genrr' #choose among 'bitflip', 'genrr', 'lapu', 'disclapu'
statistic  = 'elltwo' #choose among 'chi', 'projchi', 'elltwo'. chi requires 1-dimensional multinomial data.
n_test        = 2000        
test_start    = 1
significance_level = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############################################################################



priv_mech  = 'bitflip' #fixed in the paper
statistic  = 'projchi' #choose between 'projchi' and 'elltwo'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.get_num_threads())
data_gen = data_generator()
LDPclient = client()
significance_level = 0.05
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
    power=param_dist
    p1, p2 = create_power_law(d, power, power_2)

    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p1, sample_size),
            k,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p2, sample_size),
            k,
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
    data_entry = (test_num, d, param_dist, privacy_level, sample_size, statistic, priv_mech, statistic_vec[i].item(), p_value_vec[i].item(), float(t_end_i), time_now, n_bin, n_permutation)
    print(data_entry)

    print(f"pval: {p_value_vec[i]} -- {test_num}th test, time elapsed {t_end_i} -- emperical power so far (from test_start): {(p_value_vec[0:(i+1)] < significance_level).mean()}")

