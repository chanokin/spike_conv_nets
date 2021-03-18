import spynnaker8 as sim
import numpy as np
import matplotlib.pyplot as plt

sim.setup(timestep=1.)

np.random.seed(13)
n_neurons = 10
n_changes = 5
duration = 50.0  # ms
rates = np.random.randint(0, 5, size=(n_neurons, n_changes))*10
durations = np.ones((n_neurons, n_changes)) * duration
starts = np.repeat([np.arange(n_changes) * duration],
                   n_neurons, axis=0)
pop = sim.Population(n_neurons,  # number of sources
                     sim.extra_models.SpikeSourcePoissonVariable,
                     # source type
                     {'rates': rates,
                      'starts': starts,
                      'durations': durations
                      },  # source spike times
                     label="MF",
                     additional_parameters={'seed': 24534}
                     )

pop.record('spikes')

sim_time = duration * n_changes * 1.1

sim.run(sim_time)
neo = pop.get_data()

spikes = neo.segments[0].spiketrains

sim.end()

print(spikes)

plt.figure()
for i, spks in enumerate(spikes):
    plt.plot(spks, i * np.ones_like(spks), '.b', markersize=1.)

print(rates)

plt.show()


