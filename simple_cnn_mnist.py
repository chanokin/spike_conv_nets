import spynnaker8 as sim
import numpy as np

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)

order = data['order']
ml_conns = data['conns'].item()
ml_param = data['params'].item()

print(data.keys())


sim.setup(timestep=1.)

np.random.seed(13)
n_neurons = 10
rates = np.random.randint(0, 20, size=())
MF_population = sim.Population(num_MF_neurons,  # number of sources
                               sim.extra_models.SpikeSourcePoissonVariable,  # source type
                               {'rates': all_mf_rates,
                                'starts': all_mf_starts,
                                'durations': all_mf_durations
                                },  # source spike times
                               label="MF",
                               additional_parameters={'seed': 24534}
                               )