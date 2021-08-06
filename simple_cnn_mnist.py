import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt
import plotting
import sys
import h5py
import os
import field_encoding as fe
from field_encoding import ROWS_AS_MSB
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from pyNN.space import Line, Grid2D

H, W = 0, 1
ROWS, COLS = H, W

# def num_and_bits(shape):
#     bits = np.ceil(np.log2(shape)).astype('int')
#     num = np.power(2, np.sum(bits)).astype('int')
#     return num, bits
#
#
# def encode_with_field(msb, lsb, shift):
#     mask = (1 << shift) - 1
#     return np.bitwise_or(np.left_shift(msb, shift),
#                          np.bitwise_and(lsb, mask))


# def id_convert(ids, shape, most_significant_rows):
#     # shape = n_rows, n_cols
#     num, bits = num_and_bits(shape)
#     # extract coordinates from standard column major format
#     rows, cols = ids // shape[1], ids % shape[1]
#     # shift by n_cols if most_significant_rows == True
#     shift = bits[0] if most_significant_rows else bits[1]
#     # choose to shift rows and mask cols if most_significant_rows == True
#     msb, lsb = (rows, cols) if most_significant_rows else (cols, rows)
#
#     xy_ids = encode_with_field(msb, lsb, shift)
#
#     return xy_ids


def run_network(start_char, n_digits, n_test=10000):

    most_significant_rows = ROWS_AS_MSB

    filename = "simple_cnn_network_elements.npz"

    data = np.load(filename, allow_pickle=True)
    thresholds = dict(
        conv2d=1,#3.1836495399475098,
        conv2d_1=1,#2.9346282482147217,
        dense=1,#1.1361589431762695,
        dense_1=1,#2.435835599899292,
        dense_2=1,#2.36885929107666,
    )

    order0 = data['order']
    order = order0[:]
    ml_conns = data['conns'].item()
    ml_param = data['params'].item()

    # print(list(data.keys()))
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
#     random_state = check_random_state(0)
#     permutation = random_state.permutation(X.shape[0])
#     X = X[permutation]
#     y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    
    train_samples = 5000
    X_train, X_test, y_train, y_test = train_test_split(
                                X, y, train_size=train_samples, test_size=10000)
    
    

    test_X = X_test[start_char: start_char + n_digits]
    test_y = y_test[start_char: start_char + n_digits]
        
    # shape of dataset
    # print('X_test:  ' + str(test_X.shape))
    # print('Y_test:  ' + str(test_y.shape))

    # plotting
    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))
    #
    # plt.show()

    # MAX_N_DENSE = 128
    # MAX_N_CONV = 512
    # # sim.extra_models.SpikeSourcePoissonVariable.set_model_max_atoms_per_core(512)
    # sim.IF_curr_exp_conv.set_model_max_atoms_per_core(MAX_N_CONV)
    # sim.NIF_curr_delta.set_model_max_atoms_per_core(64)
    # # sim.IF_curr_delta_conv.set_model_max_atoms_per_core(n_atoms=256)
    # sim.IF_curr_exp_pool_dense.set_model_max_atoms_per_core(MAX_N_DENSE)
    # sim.NIF_curr_exp_pool_dense.set_model_max_atoms_per_core(MAX_N_DENSE)

    sim.setup(timestep=1.)
    sim.set_number_of_neurons_per_core(sim.NIF_curr_delta, (32, 16))

    np.random.seed(13)

    # shapes are specified as Height, Width == Rows, Columns
    shape_in = np.asarray([28, 28])
    n_in = int(np.prod(shape_in))
    in_ids = np.arange(0, n_in)
    # xy_in_ids = in_ids
    # n_in = fe.max_coord_size(shape=shape_in, most_significant_rows=ROWS_AS_MSB)
    xy_in_ids = fe.convert_ids(in_ids, shape=shape_in, most_significant_rows=ROWS_AS_MSB)
    small = np.where(xy_in_ids < np.prod(shape_in))
    # in_ids = in_ids[small]
    xy_in_ids = xy_in_ids[small]

    digit_duration = 500.0  # ms
    digit_rate = 100.0  # hz
    in_rates = np.zeros((n_in, n_digits))
    for i in range(n_digits):

        in_rates[xy_in_ids, i] = (test_X[i].flatten())[small]

    in_rates *= (digit_rate / in_rates.max())
    in_durations = np.ones((n_in, n_digits)) * np.round(digit_duration * 0.9)
    in_starts = np.repeat([np.arange(n_digits) * digit_duration],
                          n_in, axis=0)
    in_params = {
        'rates': in_rates,
        'starts': in_starts,
        'durations': in_durations
    }
    pops = {
        'input': [sim.Population(  # put into list for ease of connection
            n_in,  # number of sources
            sim.extra_models.SpikeSourcePoissonVariable,  # source type
            in_params,
            structure=Grid2D(shape_in[W] / shape_in[H]),
            label='mnist',
            additional_parameters={'seed': 24534}
        )]
    }
    sizes = {'input': [p.size for p in pops['input']]}

    # ------------------------------------------------------------------- #
    # ------------------------------------------------------------------- #
    def_params = {
        'v_thresh': 1.,
        'v_reset': 0.,
        'v': 0.,
        # 'v_rest': 0.,
        # 'tau_m': 10.,
        # 'cm': 0.80,
    }
    local_thresh = bool(0)
    use_lif = bool(0)
    cell_type = sim.IF_curr_exp if use_lif else sim.NIF_curr_delta
    # dense_cell_type = sim.IF_curr_exp_dense if use_lif else sim.NIF_curr_delta
    cell_type_dense = sim.NIF_curr_delta
    # sim.set_number_of_neurons_per_core(cell_type_dense, (32))

    for i, o in enumerate(order):
        if i == 0:
            continue

        par = ml_param[o]
        if 'conv2d' in o.lower():
            shape = par['shape'][0:2]
            chans = par['shape'][2]
            ps = def_params.copy()
            v = ps.pop('v')
            ps['v_thresh'] = thresholds[o] if local_thresh else par['threshold']
            n = int(np.prod(shape))
            # n = fe.max_coord_size(shape=shape, most_significant_rows=ROWS_AS_MSB)
            # print(o, n, shape, chans)
            pop = [sim.Population(n, cell_type, ps,
                                  structure=Grid2D(shape[W] / shape[H]),
                                  label="{}_chan_{}".format(o, ch))
                   for ch in range(chans)]

            for p in pop:
                p.set(v=v)
        elif 'dense' in o.lower():
            shape = par['shape'][0:2]
            if len(shape) == 1:
                shape = (1, shape[0])
            ps = def_params.copy()
            v = ps.pop('v')
            ps['v_thresh'] = thresholds[o] if local_thresh else par['threshold']

            # TODO: should this be converted to XY encoding as well?
            #       at this point I think any topology is lost
            n = int(np.prod(shape))
            chans = 1
            # TODO: Before I was manually splitting the flatten / dense
            #       region. Hopefully, with the automatic splitting, we can
            #       get all the network to fit in the small board
            # if 'conv2d' in order[i-1]:
            #     chans = 4
            #     n = n // chans

            pop = [sim.Population(n, cell_type_dense, ps,
                                  structure=Grid2D(shape[W]/shape[H]),
                                  label="{}_chan_{}".format(o, ch))
                   for ch in range(chans)]
            for p in pop:
                p.set(v=v)

        sizes[o] = [p.size for p in pop]
        pops[o] = pop

    rec = [
        'input',
        'conv2d',
        'conv2d_1',
        'dense',
        'dense_1',
        'dense_2',
    ]

    shapes = {
        'input': [28, 28],
        'conv2d': [24, 24],
        'conv2d_1': [8, 8],
        'dense': [12, 12],
        'dense_1': [8, 8],
        'dense_2': [4, 4],
    }

    offsets = {
        'input': 0,
        'conv2d': 0,
        'conv2d_1': 0,
        'dense': 32,
        'dense_1': 0,
        'dense_2': 0,
    }

    for k in rec:
        for p in pops[k][:]:
            p.record('spikes')

    projs = {}
    kernels = {}
    dense_weights = {}

    def norm_w(w, is_conv=False, trans=None):
        new_w = w.copy()
        if trans == 'linear':
            max_w = np.max(np.abs(w))
            new_w /= max_w
        elif trans == 'sum_to_0':
            pos = w[w > 0]
            pos /= np.sum(pos)
            neg = w[w < 0]
            neg /= (-np.sum(neg))
            new_w = w.copy()
            new_w[w > 0] = pos
            new_w[w < 0] = neg
        elif trans == 'mean_var':
            v = new_w.var()
            new_w -= new_w.mean()
            new_w /= v

        return new_w

    for i, o in enumerate(order):
        if i == 0:
            continue
        c = ml_conns[o]
        weights = c['weights']
        pooling = 'pool' in c
        pool_area = np.asarray(c['pool']['shape']) if pooling else None
        pool_stride = np.asarray(c['pool']['strides']) if pooling else None
        wshape = c.get('shape', None)
        strides = c.get('strides', None)

        o0 = order[i-1]
        # dense_weights[o] =
        pops0 = pops[o0]
        pops1 = pops[o]
        # print(o0, o)
        for prei, pre in enumerate(pops0):
            pre_shape = np.asarray(ml_param[o0]['shape'][:2])
            if len(pre_shape) == 1:
                pre_shape = (1, pre.size)
                n_chan = 1
            else:
                n_chan = ml_param[o0]['shape'][-1]

            wl = []
            for posti, post in enumerate(pops1):
                lbl = "{}_{} to {}_{}".format(o0, prei, o, posti)
                # print(pre_shape, n_chan, o0, prei, o, posti)
                if 'conv2d' in o.lower():
                    # print(prei, posti, wshape, c['weights'].shape)
                    w = norm_w(weights[:, :, prei, posti].copy())
                    # note we need to flip kernels for the operation to be a
                    # convolution instead of a correlation
                    w = np.flipud(np.fliplr(w))
                    wl.append(w)
                    cn = sim.ConvolutionConnector(w, strides=strides,
                                                  pool_shape=pool_area,
                                                  pool_stride=pool_stride,
                                                  )
                    prj = sim.Projection(pre, post, cn, sim.Convolution(), label=lbl)
                    projs[lbl] = prj

                elif 'dense' in o.lower():
                    n_out = post.size
                    sh_pre = sim.PoolDenseConnector.get_post_pool_shape(
                                            pre_shape, pool_area, pool_stride)
                    size_pre = int(np.prod(sh_pre))
                    if 'conv2d' in o0.lower():
                        pre_is_conv = True
                        cnv = pops[o0]
                        col0 = posti * n_out
                        col1 = col0 + n_out

                        mtx_rows = []
                        chan = prei
                        for r in np.arange(sh_pre[0]):
                            for c in np.arange(sh_pre[1]):
                                mtx_rows.append(r * sh_pre[1] * len(pops[o0]) +
                                                c * len(pops[o0]) + chan)
                        mtx_rows = np.asarray(mtx_rows)
                        pre_rows = np.repeat(np.arange(sh_pre[0]), sh_pre[1])
                        # print("pre_rows = {}".format(pre_rows))
                        pre_cols = np.tile(np.arange(sh_pre[1]), sh_pre[0])
                        # print("pre_cols = {}".format(pre_cols))
                        mtx_rows0 = (pre_rows * sh_pre[1] * n_chan +
                                     pre_cols * n_chan + prei)
                        # print("mtx_rows = {}".format(mtx_rows))
                        # print(np.all(mtx_rows == mtx_rows0))
                        n_rows = len(mtx_rows)
                        mtx_rows = np.repeat(mtx_rows, n_out)
                        # print("mtx_rows = {}".format(mtx_rows))
                        mtx_cols = np.arange(col0, col1)
                        # print("mtx_cols = {}".format(mtx_cols))
                        mtx_cols = np.tile(mtx_cols, n_rows)
                        # print("mtx_cols = {}".format(mtx_cols))
                        ws = weights[mtx_rows, mtx_cols].copy().reshape((n_rows, n_out))
                        # print(ws.shape)
                        # print(ws)
                        # row0 = prei * size_pre
                        # row1 = row0 + size_pre
                        # ws = weights[row0:row1, col0:col1]

                    else:
                        pre_is_conv = False
                        row0 = prei * size_pre
                        row1 = row0 + size_pre
                        ws = weights[row0:row1, :]

                    # for cidx in range(ws.shape[1]):
                    #     ws[:, cidx] = norm_w(ws[:, cidx])

                    wl.append(ws)
                    cn = sim.PoolDenseConnector(ws, pool_area, pool_stride)

                    prj = sim.Projection(pre, post, cn, sim.PoolDense(), label=lbl)
                    projs[lbl] = prj

            kernels[o] = wl

    sim_time = digit_duration #* (n_digits + 0.1)
    all_neos = []
    all_spikes = []
    for ch_idx in range(n_digits):
        print("--------------- character {} ---------------".format(
            ch_idx + start_char))
        neos = {}
        spikes = {}

        # sim.reset()

        sim.run(sim_time)

        for k in rec:
            neos[k] = [p.get_data() for p in pops[k]]
            spikes[k] = [x.segments[0].spiketrains for x in neos[k]]

        all_neos.append(neos)
        all_spikes.append(spikes)

        # sim.reset()

        for k in pops:
            if 'conv' in k or 'dense' in k:
                for p in pops[k]:
                    p.set(v=0)

    sim.end()

    with h5py.File("output_data_simple_cnn_mnist.h5", "a") as h5:
        # sim.reset()
        smp = "sample"
        tgt = "target"
        rts = "rates"
        nt = "n_test"
        if not smp in h5:
            gsamp = h5.create_dataset(smp, (10000, 1), dtype='int')
            gtgt = h5.create_dataset(tgt, (10000, 1), dtype='int')
            grts = h5.create_dataset(rts, (10000, 10), dtype='int')
            gnt = h5.create_dataset(nt, (1,), dtype='int')
        else:
            gsamp = h5[smp]
            gtgt = h5[tgt]
            grts = h5[rts]
            gnt = h5[nt]

        # for ch_idx in range(n_digits):
        #     aidx = ch_idx + start_char
        #     rs = [len(ts) for ts in all_spikes[ch_idx]['dense_2'][0]]
        #     gnt[:] = aidx + 1
        #     gsamp[aidx, 0] = aidx
        #     gtgt[aidx, 0] = test_y[ch_idx]
        #     grts[aidx, :] = rs
        #
        #     ty = test_y[ch_idx]
        #     py = np.argmax(rs)
        #
        #     print("Sample {}\tPredicted = {}\tExpected = {}".format(aidx, py, ty))

    import plot_simple_cnn_mnist as splt

    # for i, _spikes in enumerate(all_spikes):
    prefix = "{:03}".format(start_char)
#     prefix = "{:03}".format(0)
    data = splt.plot_images(order, shapes, test_y, kernels, spikes,
                            n_digits*sim_time, digit_duration, offsets, norm_w,
                            n_digits, prefix)
    rates, conf_mtx, correct, no_spikes = data

    splt.plot_matrix(conf_mtx, n_digits, no_spikes, correct, prefix)
    splt.plot_rates(rates, order, prefix=prefix)
    splt.plot_spikes(order, spikes, sim_time, digit_duration, prefix)

    np.savez_compressed('activity_for_simple_cnn.npz',
                        order=order, shapes=shapes, test_y=test_y, kernels=kernels,
                        spikes=spikes, total_sim_time=n_digits*sim_time,
                        digit_duration=digit_duration, offsets=offsets,
                        n_digits=n_digits, rates=rates, conf_mtx=conf_mtx,
                        correct=correct, no_spikes=no_spikes)
    # plt.show()

if __name__ == '__main__':
    start_char = int(sys.argv[1])
    n_digits = int(sys.argv[2])
    n_test = int(sys.argv[3])
    print(
        "======================================================="
        "\n start character index {} "
        "\n number of characters per run {}"
        "\n=======================================================\n\n".format(
            start_char, n_digits))
    run_network(start_char, n_digits)