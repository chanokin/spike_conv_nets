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
for layer in recs:
    spikes = recs[layer][0].segments[0].spiketrains
    print(layer)
    print(spikes)