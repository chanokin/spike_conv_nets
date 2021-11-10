import numpy as np
import matplotlib.pyplot as plt
import plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable


def to_dict(np_file):
    d = {}
    for k in np_file.keys():
        try:
            d[k] = np_file[k].item()
        except:
            d[k] = np_file[k]
    return d

_colors = ['red', 'green', 'blue', 'cyan', 'orange',
           'magenta', 'black', 'yellow', ]

input_filename = "Bifrost_Network_recordings.npz"
np_data = np.load(input_filename, allow_pickle=True)
data_dict = to_dict(np_data)

recs = data_dict['recordings']
shapes = data_dict['shapes']
in_cfg = data_dict['input_configuration']
n_samples = in_cfg['num_samples']
on_time = in_cfg['on_time_ms']
off_time = in_cfg['off_time_ms']
classes = in_cfg['target_classes']
period = on_time + off_time
run_time = period * n_samples
for layer_idx, layer in enumerate(recs):

    shape = shapes[layer]
    for channel in recs[layer]:
        print(layer, channel)
        n_rows = 1
        figsize = np.array([n_samples, n_rows]) * 5.

        spikes = recs[layer][channel].segments[0].spiketrains
        voltages = recs[layer][channel].segments[0].filter(name='v')

        images = plotting.spikes_to_images(spikes, shape, run_time, period)
        max_sh = np.max(shape)
        prod_sh = np.prod(shape)

        if max_sh == prod_sh:
            size = int(np.prod(shape))
            rows = int(np.round(np.sqrt(size)))
            cols = size // rows + int(size % rows > 0)
            new_shape = [rows, cols]
        else:
            new_shape = shape

        new_size = int(np.prod(new_shape))
        fig, axs = plt.subplots(n_rows, n_samples, sharey=True, figsize=figsize)
        plt.suptitle(f"spynn {layer}, {channel}")
        for i, img in enumerate(images[0]):
            ax = axs if n_samples == 1 else axs[i]
            ax.set_title(classes[i])
            new_image = np.zeros(new_size)
            new_image[:img.size] = img.flatten()
            im = ax.imshow(new_image.reshape(new_shape))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.savefig(f"images_layer_{layer_idx:03d}_{layer}_channel_{channel:03d}.png", dpi=150)
        plt.close(fig)

        # fig, ax = plt.subplots(1, 1)
        # plt.suptitle(f"raster {layer}, {channel}")
        # for neuron_idx, times in enumerate(spikes):
        #     ax.plot(times, neuron_idx * np.ones_like(times), '.b', markersize=1)
        # plt.savefig(f"raster_layer_{layer_idx:03d}_{layer}_channel_{channel:03d}.png", dpi=150)
        # plt.close(fig)


        if len(voltages):
            fig, ax = plt.subplots(1, 1)
            plt.suptitle(f"spynn {layer}, {channel} volts")
            ax.plot(voltages[0])
            plt.savefig(f"voltages_layer_{layer_idx:03d}_{layer}_channel_{channel:03d}.png", dpi=150)
            plt.close(fig)

            fig, axs = plt.subplots(n_rows, n_samples, sharey=True, figsize=figsize)
            plt.suptitle(f"spynn {layer}, {channel}")
            for v_idx, stt in enumerate(np.arange(0, n_samples * period, period)):
                stt = int(stt)
                end = stt + int(period)
                ax = axs if n_samples == 1 else axs[v_idx]
                vs = np.zeros(new_size)
                vs[:voltages[0].shape[1]] = np.sum(voltages[0][stt:end,:], axis=0)
                # ax.plot(voltages[0], linewidth=0.1)
                im = ax.imshow(vs.reshape(new_shape))
                ax.set_title(classes[v_idx])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

            plt.savefig(f"voltages_sum_layer_{layer_idx:03d}_{layer}_channel_{channel:03d}.png", dpi=150)
            plt.close(fig)
    # print(spikes)