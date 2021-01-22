import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import matplotlib.pyplot as plt

np.random.seed(13)

sim.SpikeSourceArray.set_model_max_atoms_per_core(n_atoms=11)

shape = np.array([5, 5], dtype='int32')  # h, w
n_input = np.prod(shape, dtype='int32')
stride = np.array([1, 1], dtype='int32')  # h, w
k_shape = np.array([3, 3], dtype='int32')
vline = [[20.+np.random.randint(-2, 3)]
         if (idx % shape[1]) == (shape[1] // 2) or idx == 13 else []
         for idx in range(n_input)]


wmax = 5.0
ws = np.random.uniform(-wmax, wmax, k_shape)
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

src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline}, label='input spikes')

conn = sim.ConvolutionConnector(shape, ws, strides=stride)
shape_out = conn.get_post_shape()
sum_inputs = np.zeros(shape_out)
hh, hw = k_shape // 2
for i, x in enumerate(vline):
    if len(x):
        print(i, x)
        r, c = i // shape[1], i % shape[1]
        print("pre {}, r {}, c {}".format(i, r, c))
        postr, postc = conn.pre_as_post(r, c)
        print("postr {}, postc {}".format(postr, postc))
        for kr in range(-hh, hh+1):
            for kc in range(-hw, hw+1):
                newr = postr + kr
                newc = postc + kc
                if (newr < 0 or newr >= shape_out[0] or
                    newc < 0 or newc >= shape_out[1]):
                    continue
                www = ws[kr + k_shape[0] // 2, kc + k_shape[1] // 2]
                print("row {}, col {}, w {}".format(newr, newc, www))

                sum_inputs[newr, newc] += www

print(sum_inputs)

n_out = np.prod(shape_out, dtype='int32')
post_cfg = {
    'v_thresh': -60.0,
    'v_reset': -80.0,
    'v_rest': -65.0
}

dst = sim.Population(
        n_out, sim.IF_curr_exp_conv, post_cfg)
dst.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)

prj = sim.Projection(src, dst, conn)

sim.run(run_time)

neo = dst.get_data('v')
v = neo.segments[0].filter(name='v')[0]
spikes = neo.segments[0].spiketrains
print(v)
print(spikes)

sim.end()

sum_inputs += post_cfg['v_rest']
maxv = max(post_cfg['v_thresh']*0.9, np.max(sum_inputs))
color = ['red', 'blue', 'green', 'orange']
plt.figure()
plt.axhspan(post_cfg['v_thresh'], maxv, color='gray', alpha=0.3)
plt.axhline(post_cfg['v_reset'], color='gray', linestyle=':')
for i, w in enumerate(sum_inputs.flatten()):
    plt.axhline(w, color=color[i], linestyle='--')

for i, vv in enumerate(v.T):
    plt.plot(vv, color=color[i], label=i)

plt.legend()
plt.show()

