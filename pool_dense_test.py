import numpy as np
import spynnaker8 as sim
from pyNN.space import Grid2D
import matplotlib.pyplot as plt
import field_encoding as fe

np.random.seed(13)

if bool(1):
    sim.SpikeSourceArray.set_model_max_atoms_per_core(10)
    # sim.SpikeSourcePoisson.set_model_max_atoms_per_core(32)
    sim.IF_curr_exp_pool_dense.set_model_max_atoms_per_core(16)

n_pre = 2
ROWS_ARE_MSB = bool(1)
pre_is_conv = bool(0)
if pre_is_conv:
    shape = np.array([5, 5], dtype='int32')  # h, w
    stride = np.array([1, 1], dtype='int32')  # h, w
    k_shape = np.array([3, 3], dtype='int32')

    n_input = fe.max_coord_size(shape=shape,
                                most_significant_rows=ROWS_ARE_MSB)
else:
    shape = np.array([1, 13], dtype='int32')
    n_input = 13

# vline = [[20.+np.random.randint(-2, 3)]
vline = [[20. + idx // shape[1]]
         if (idx % shape[1]) == (shape[1] // 2) else []
         for idx in range(n_input)]

vline0 = [[10. + idx // shape[1]]
         if (idx % shape[1]) == (shape[1] // 2) else []
         for idx in range(n_input)]
# vline = [[20. + idx]
#          if ((idx % shape[1]) == (shape[1] // 2) and
#              (idx % shape[0]) == (shape[0] // 2))
#          else []
#          for idx in range(n_input)]


k_shape = (16, 16)
wmax = 5.0
# ws = np.random.uniform(-wmax, wmax, k_shape)
# ws = np.arange(int(np.prod(k_shape))).reshape(k_shape)
# ws = np.arange(np.prod(k_shape), dtype='float').reshape(k_shape)
# ws[:, k_shape[1]//2] = np.arange(k_shape[1]) + 2.
# ws[:, k_shape[1]//2] *= -0.8
# ws[:, k_shape[1]//2+1:] = 0.
# print(ws)
# print(np.sum(ws))
# ws[:] = ws / np.sum(ws**2)
# ws[:] = ws - np.mean(ws)
# ws[ws > 0] /= np.sum( (ws > 0) * ws )
# ws[ws < 0] /= -np.sum( (ws < 0) * ws )
# ws *= 3.
# print(np.sum(ws))
# print(ws)



run_time = 60.

sim.setup(timestep=1.)
src = sim.Population(n_input, sim.SpikeSourceArray,
                     {'spike_times': vline},
                     label='input spikes 0')

src1 = sim.Population(n_input, sim.SpikeSourceArray,
                      {'spike_times': vline0},
                      label='input spikes 1')

pooling = np.asarray([2, 2]) if pre_is_conv else 1
pooling_stride = np.asarray([2, 2])
pool_shape = sim.PoolDenseConnector.calc_post_pool_shape(
                                    shape, pre_is_conv, pooling, pooling_stride)
n_out = 23
k_shape = np.asarray(
    (int(np.prod(pool_shape)), n_out),
    dtype='int')

div = 1. / np.prod(pooling_stride)
ws = np.arange(int(np.prod(k_shape))).reshape(k_shape) * 0.01
print()
print(ws)
print(np.max(ws))
print(np.max(ws * div))
print()

conn = sim.PoolDenseConnector(0, shape, ws, n_out, pooling, pooling_stride,
                              pre_is_conv=pre_is_conv)
conn1 = sim.PoolDenseConnector(0, shape, ws - 1.0, n_out, pooling, pooling_stride,
                               pre_is_conv=pre_is_conv)


post_cfg = {
    'v_thresh': -60.0,
    'v_reset': -80.0,
    'v_rest': -65.0
}

dst = sim.Population(
        n_out, sim.IF_curr_exp_pool_dense, post_cfg)
dst.record(['v', 'spikes'])
# syn = sim.StaticSynapse(weight=ws.flatten)

prj = sim.Projection(src, dst, conn)
prj1 = sim.Projection(src1, dst, conn1)

sim.run(run_time)

neo = dst.get_data()
v = neo.segments[0].filter(name='v')[0]
spikes = neo.segments[0].spiketrains
print(v)
print(spikes)

sim.end()

# sum_inputs += post_cfg['v_rest']
# maxv = max(post_cfg['v_thresh']*0.9, np.max(sum_inputs))
# color = ['red', 'blue', 'green', 'orange']
#
# plt.figure()
# plt.axhspan(post_cfg['v_thresh'], maxv, color='gray', alpha=0.3)
# plt.axhline(post_cfg['v_reset'], color='gray', linestyle=':')
# for i, w in enumerate(sum_inputs.flatten()):
#     plt.axhline(w, linestyle='--')#, color=color[i])
#
# for i, vv in enumerate(v.T):
#     plt.plot(vv, label=i)#, color=color[i])
#
# for i, spks in enumerate(spikes):
#     for t in spks:
#         plt.axvline(float(t), linestyle=':')#, color=color[i])
#
# plt.legend()
#
# plt.show()
#
