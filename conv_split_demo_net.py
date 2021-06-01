import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import matplotlib.pyplot as plt

np.random.seed(13)
neuron_type = sim.IF_curr_exp_conv
sim.SpikeSourceArray.set_model_max_atoms_per_core(n_atoms=18)
neuron_type.set_model_max_atoms_per_core(n_atoms=18)
n_sources = 1
# rate = 200
shape = np.array([7, 7], dtype='int32')  # h, w
bits_w = int(np.ceil(np.log2(shape[1])))
bits_h = int(np.ceil(np.log2(shape[0])))
n_input = 2**(bits_h + bits_w)  # int(np.prod(shape, dtype='int32'))
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([3, 3], dtype='int32')
# vline = [[20.+np.random.randint(-2, 3)]
# vline = [[20.]
#          if (idx % shape[1]) == (shape[1] // 2) else []
#          for idx in range(n_input)]
n_in_spikes = 40
vline = [[20.]
         if idx < n_in_spikes else []
         for idx in range(n_input)]

wmax = 5.0
# ws = np.random.uniform(-wmax, wmax, k_shape)
ws = (np.arange(np.prod(k_shape)).reshape(k_shape) * 0.25 - 2.5) * 0.1
# ws = np.arange(np.prod(k_shape), dtype='float').reshape(k_shape)
# ws[:, k_shape[1]//2] = np.arange(k_shape[1]) + 2.
# ws[:, k_shape[1]//2] *= -0.8
# ws[:, k_shape[1]//2+1:] = 0.
print(ws)
print(np.sum(ws))
# ws[:] = ws / np.sum(ws**2)
# ws[:] = ws - np.mean(ws)
# ws[ws > 0] /= np.sum( (ws > 0) * ws )
# ws[ws < 0] /= -np.sum( (ws < 0) * ws )
# ws *= 3.
# print(np.sum(ws))
# print(ws)

run_time = 50.

sim.setup(timestep=1.)

src = [sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline},
                      label='input spikes {}'.format(i))
       for i in range(n_sources)]

for s in src:
    s.record('spikes')
# src = [sim.Population(n_input, sim.SpikeSourcePoisson,
#                      {'rate': rate},
#                       label='input spikes {}'.format(i))
#        for i in range(n_sources)]
# for s in src:
#     s.record('spikes')

conn = sim.ConvolutionConnector(shape, ws, strides=stride)
shape_out = conn.get_post_shape()
sum_inputs = np.zeros(shape_out)
hh, hw = k_shape // 2
# for i, x in enumerate(vline):
#     if len(x):
#         print(i, x)
#         r, c = i // shape[1], i % shape[1]
#         print("pre {}, r {}, c {}".format(i, r, c))
#         postr, postc = conn.pre_as_post(r, c)
#         print("postr {}, postc {}".format(postr, postc))
#         for kr in range(-hh, hh+1):
#             for kc in range(-hw, hw+1):
#                 newr = postr + kr
#                 newc = postc + kc
#                 if (newr < 0 or newr >= shape_out[0] or
#                     newc < 0 or newc >= shape_out[1]):
#                     continue
#                 www = ws[kr + k_shape[0] // 2, kc + k_shape[1] // 2]
#                 print("row {}, col {}, w {}".format(newr, newc, www))
#
#                 sum_inputs[newr, newc] += www
#
# print(sum_inputs)

n_out = int(np.prod(shape_out, dtype='int32'))
bit_w = int(np.ceil(np.log2(shape_out[1])))
bit_h = int(np.ceil(np.log2(shape_out[0])))
n_out = 2**(bit_w + bit_h)

post_cfg = {
    'v_thresh': -60.0,
    'v_reset': -80.0,
    'v_rest': -65.0
}

dst = sim.Population(n_out, neuron_type, post_cfg)
# dst.record(['v', 'spikes'])
dst.record('spikes')
# syn = sim.StaticSynapse(weight=ws.flatten)

prj = [sim.Projection(src[i], dst, conn) for i in range(n_sources)]

sim.run(run_time)

neo = dst.get_data()
# v = neo.segments[0].filter(name='v')[0]
spikes = neo.segments[0].spiketrains
# print(v)
# print(spikes)

in_neos = [p.get_data() for p in src]
in_spikes = [n.segments[0].spiketrains for n in in_neos]

sim.end()

colors = ['red', 'green', 'blue', 'cyan', 'orange',
          'magenta', 'black', 'yellow', ]
fig = plt.figure(figsize=(16, 4))
ax = plt.subplot(1, 2, 1)
for i, s in enumerate(in_spikes):
    for j, ts in enumerate(s):
        ts = np.asarray([t for t in ts])
        ax.plot(ts + 0.1 * i, j * np.ones_like(ts), '.',
                markersize=1., color=colors[i], alpha=0.5)

ax = plt.subplot(1, 2, 2)
for j, ts in enumerate(spikes):
    ax.plot(ts, j * np.ones_like(ts), '.k',
            markersize=1.)

plt.show()