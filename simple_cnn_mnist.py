import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)

order = data['order']
ml_conns = data['conns'].item()
ml_param = data['params'].item()

print(list(data.keys()))

test_X = mnist.test_images()
test_y = mnist.test_labels()

# shape of dataset
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# plotting

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))

plt.show()
sim.extra_models.SpikeSourcePoissonVariable.set_model_max_atoms_per_core(100)
sim.setup(timestep=1.)

np.random.seed(13)

shape_in = np.asarray([28, 28])
n_in = int(np.prod(shape_in))
n_digits = 5
digit_duration = 500.0  # ms
digit_rate = 5.0  # hz
in_rates = np.zeros((n_in, n_digits))
for i in range(n_digits):
    in_rates[:, i] = test_X[i].flatten()

in_rates *= (digit_rate / in_rates.max())
in_durations = np.ones((n_in, n_digits)) * digit_duration
in_starts = np.repeat([np.arange(n_digits) * digit_duration],
                      n_in, axis=0)
in_params = {
    'rates': in_rates,
    'starts': in_starts,
    'durations': in_durations
}
pops = {
    'in': sim.Population(
        n_in,  # number of sources
        sim.extra_models.SpikeSourcePoissonVariable,  # source type
        in_params,
        label='mnist',
        additional_parameters={'seed': 24534}
    )
}

pops['in'].record('spikes')

sim_time = digit_duration * n_digits * 1.1

sim.run(sim_time)
neo = pops['in'].get_data()

spikes = neo.segments[0].spiketrains

sim.end()

# print(spikes)

plt.figure()
for i, spks in enumerate(spikes):
    plt.plot(spks, i * np.ones_like(spks), '.b', markersize=1.)
plt.show()


