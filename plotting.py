import matplotlib.pyplot as plt
import numpy as np

def spikes_to_bins(spikes, max_t, t_bin):
    tbins = np.arange(0, max_t, t_bin)
    sbins = [[[] for _ in spikes] for _ in tbins]
    for nidx, spks in enumerate(spikes):
        if len(spks):
            ids = np.digitize(np.asarray(spks), tbins)
            for sidx, bidx in enumerate(ids):
                sbins[bidx-1][nidx].append(spks[sidx])

    return sbins

def spikes_to_images(spikes, shape, max_t, t_bin):
    binned = spikes_to_bins(spikes, max_t, t_bin)
    images = []
    for bidx, sbin in enumerate(binned):
        img = np.zeros(shape)
        for nidx, spks in enumerate(sbin):
            nspks = len(spks)
            if nspks:
                r, c = nidx // shape[1], nidx % shape[1]
                img[r, c] = nspks
        images.append(img)

    return images, binned


