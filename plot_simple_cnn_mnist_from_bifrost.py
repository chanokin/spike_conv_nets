import numpy as np
import matplotlib.pyplot as plt
import plotting
import field_encoding as fe


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
period = on_time + off_time
run_time = period * n_samples
for layer in recs:
    spikes = recs[layer][0].segments[0].spiketrains
    images = plotting.spikes_to_images(spikes, shapes[layer], run_time, period)
    fig, axs = plt.subplots(1, n_samples, sharey=True)
    plt.suptitle(layer)
    for i, img in enumerate(images[0]):
        axs[i].imshow(img)
    plt.savefig(f"images_layer_{layer}.png", dpi=150)

    # print(spikes)