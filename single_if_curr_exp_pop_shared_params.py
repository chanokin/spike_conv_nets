import numpy as np
import spynnaker8 as spynn
import sys

n_neurons = 13

spynn.setup(1.0)
pop = spynn.Population(n_neurons, spynn.IF_curr_exp,
                       {'tau_syn_E': np.random.uniform(10, 15, n_neurons)},
                       label='neurons_random_tau_syn_E')

spynn.run(1.0)
spynn.end()