import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import numpy as np
import sys
import os
from spike_conv_nets.simple_cnn_mlgenn import plotting as ptt
import cv2


def imshow(ax, img, cmap='cividis'):
    ax.imshow(img, interpolation='none', cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


test = 1 if bool(0) else None

with np.load('activity_for_simple_cnn.npz', allow_pickle=True) as f:
    print(f)
    # order = order, shapes = shapes, test_y = test_y, kernels = kernels,
    # spikes = spikes, total_sim_time = n_digits * sim_time,
    # digit_duration = digit_duration, offsets = offsets,
    # n_digits = n_digits, rates = rates, conf_mtx = conf_mtx,
    # correct = correct, no_spikes = no_spikes
    fname_template = "{}_{:03d}_{:06d}.png"
    order = f['order']
    shapes = f['shapes'].item()
    test_y = f['test_y']
    spikes = f['spikes'].item()
    total_sim_time = f['total_sim_time']
    digit_duration = f['digit_duration']
    n_digits = f['n_digits']
    frame_time = 25.
    sbins = {k: [ptt.spikes_to_bins(spks, total_sim_time, frame_time)
                 for spks in spikes[k]]
             for k in spikes}

    n_frames = len(sbins['input'][0])

    out_path = "video_out"
    os.makedirs(out_path, exist_ok=True)
    shapes['dense'] = [32, 4]
    shapes['dense_1'] = [32, 2]
    shapes['dense_2'] = [10, 1]

    imgs = []
    for layer_idx, layer_name in enumerate(sbins):
        lpath = os.path.join(out_path, layer_name)
        os.makedirs(lpath, exist_ok=True)
        for channel, sbb in enumerate(sbins[layer_name]):
            schan = str(channel)
            chpath = os.path.join(lpath, schan)
            os.makedirs(chpath, exist_ok=True)
            sys.stdout.write("\rLayer {}{}\tChannel {:02d}".format(
                layer_name, shapes[layer_name], channel
            ))
            sys.stdout.flush()

            imgs[:] = ptt.__plot_binned_spikes(sbb, shapes[layer_name], 0)
            for frame_idx, im in enumerate(imgs):
                fname = fname_template.format(layer_name, channel, frame_idx)
                maxi = np.max(im)
                if maxi > 0:
                    im *= 255. / np.max(im)
                img_path = os.path.join(chpath, fname)
                cv2.imwrite(img_path, im.astype('uint8'))
            # time.sleep(0.001)
        print()

    for frame_idx in range(n_frames)[:test]:
        sys.stdout.write("\rRendering frame {:06d}/{}".format(frame_idx, n_frames))
        sys.stdout.flush()

        layer_images = {}
        for layer_idx, layer_name in enumerate(sbins):
            lpath = os.path.join(out_path, layer_name)
            channel_images = {}
            for channel, sbb in enumerate(sbins[layer_name]):
                schan = str(channel)
                chpath = os.path.join(lpath, schan)
                fname = fname_template.format(layer_name, channel, frame_idx)

                img_path = os.path.join(chpath, fname)
                channel_images[channel] = cv2.imread(img_path)[:, :, 0].astype('float')
                # print(channel_images[channel].dtype)
            layer_images[layer_name] = channel_images

        fig = plt.figure()
        gs = gspec.GridSpec(ncols=8, nrows=8, figure=fig)
        ax = fig.add_subplot(gs[:, 0])
        imshow(ax, layer_images['input'][0])

        layer_name = 'conv2d'
        col_start = 1
        ncols = 2
        nchan = len(layer_images[layer_name])
        div = nchan // ncols
        for chidx, img in layer_images[layer_name].items():
            r = chidx % div
            c = col_start + chidx // div
            ax = fig.add_subplot(gs[r, c])
            imshow(ax, img)

        layer_name = 'conv2d_1'
        col_start += ncols
        ncols = 1
        nchan = len(layer_images[layer_name])
        div = nchan // ncols
        for chidx, img in layer_images[layer_name].items():
            r = chidx % div
            c = col_start + chidx // div
            ax = fig.add_subplot(gs[r, c])
            imshow(ax, img)

        layer_name = 'dense'
        col_start += ncols
        ncols = 1
        nchan = len(layer_images[layer_name])
        div = nchan // ncols
        for chidx, img in layer_images[layer_name].items():
            c = col_start + chidx // div
            ax = fig.add_subplot(gs[:, c])
            imshow(ax, img)

        layer_name = 'dense_1'
        col_start += ncols
        ncols = 1
        nchan = len(layer_images[layer_name])
        div = nchan // ncols
        for chidx, img in layer_images[layer_name].items():
            c = col_start + chidx // div
            ax = fig.add_subplot(gs[:, c])
            imshow(ax, img)

        layer_name = 'dense_2'
        col_start += ncols
        ncols = 1
        nchan = len(layer_images[layer_name])
        div = nchan // ncols
        for chidx, img in layer_images[layer_name].items():
            c = col_start + chidx // div
            ax = fig.add_subplot(gs[:, c])
            imshow(ax, img)
            for n in range(10):
                ax.text(-0.25, 0.25 + n, str(n), fontsize=28, color='#00224d')

        plt.tight_layout()
        fig_name = os.path.join(out_path, "frame_{:06d}.png".format(frame_idx))
        plt.savefig(fig_name, dpi=300)
        if test:
            plt.show()

        plt.close(fig)

    print("\nDone!")
