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

def spikes_to_images(spikes, shape, max_t, t_bin, offset_row=0):
    binned = spikes_to_bins(spikes, max_t, t_bin)
    images = []
    max_idx = -1
    for bidx, sbin in enumerate(binned):
        img = np.zeros(shape)
        for nidx, spks in enumerate(sbin):
            nspks = len(spks)
            if nspks:
                if nidx > max_idx:
                    max_idx = nidx
                r, c = nidx // shape[1], nidx % shape[1]
                r += offset_row
                # print(nidx, r, c, nspks)
                img[r, c] = nspks
        images.append(img)
    print("\n\n\nmax_index found for spikes {}\n\n".format(max_idx))
    return images, binned


