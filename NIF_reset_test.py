import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import cv2
import matplotlib.pyplot as plt
import sys

run_time = 50.
w = 0.3
sim.setup(timestep=1.)

src = sim.Population(1, sim.SpikeSourceArray,
                     {'spike_times': [[1, 2, 3]]}, label='input spikes')

params = {
    'v_thresh': 1.,
    'v_reset': 0.,
    'v': 0.,
}
output = sim.Population(1, sim.NIF_curr_exp_conv,
                      params, label="out")

conn = sim.ConvolutionConnector((1,1), np.array([[w]]), strides=1)

proj = sim.Projection(src, output, conn)

output.record(['v', 'spikes'])
run_time = 10.
sim.run(run_time)

neo = output.get_data()

# sim.reset()
output.set(v=0.)
src.set(spike_times=[[14, 15, 16, 17]])

sim.run(10)
neo2 = output.get_data()

print('end')
sim.end()

v0 = neo.segments[0].filter(name='v')[0]
v1 = neo2.segments[0].filter(name='v')[0]

plt.figure()
plt.plot(v0, marker='.')
plt.plot(v1, marker='+')
plt.show()
