import matplotlib.pyplot as plt
import numpy as np

import field_encoding as fe
from field_encoding import ROWS_AS_MSB


def spikes_to_bins(spikes, max_t, t_bin):
    tbins = np.arange(0, max_t, t_bin)
    sbins = [[[] for _ in spikes] for _ in tbins]
    for nidx, spks in enumerate(spikes):
        if len(spks):
            ids = np.digitize(np.asarray(spks), tbins)
            for sidx, bidx in enumerate(ids):
                sbins[bidx-1][nidx].append(spks[sidx])

    return sbins


def __plot_binned_spikes(binned, shape, offset_row=0):
    images = []
    max_idx = -1
    for bidx, sbin in enumerate(binned):
        # new_shape = fe.get_augmented_shape(shape, most_significant_rows=ROWS_AS_MSB)
        img = np.zeros(shape)
        for nidx, spks in enumerate(sbin):
            nspks = len(spks)
            if nspks > 0:
                # nidx += offset_row
                # if nidx > max_idx:
                #     max_idx = nidx

                # r, c = fe.decode_ids(nidx, shape=shape,
                #                      most_significant_rows=not ROWS_AS_MSB)
                # if r >= shape[0] or c >= shape[1]:
                #     continue

                r, c = nidx // shape[1], nidx % shape[1]

                print(shape, nidx, r, c, nspks)
                img[r, c] = nspks
        images.append(img)
    # print("\n\n\nmax_index found for spikes {}\n\n".format(max_idx))
    return images


def spikes_to_images(spikes, shape, max_t, t_bin):
    imgs, bins = spikes_to_images_list([spikes], shape, max_t, t_bin, 0, False)
    return imgs[0], bins[0]


def spikes_to_images_list(spikes_list, shape, max_t, t_bin, offset_row,
                          merge_images=False):
    bins = [spikes_to_bins(s, max_t, t_bin) for s in spikes_list]
    imgs = [__plot_binned_spikes(b, shape, offset_row * i)
            for i, b in enumerate(bins)]

    if merge_images:
        img0 = imgs[0]
        for i in range(1, len(imgs)):
            for j in range(len(img0)):
                img0[j] += imgs[i][j]
        return img0, bins

    return imgs, bins





