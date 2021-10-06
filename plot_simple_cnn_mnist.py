import numpy as np
import matplotlib.pyplot as plt
from spike_conv_nets.simple_cnn_mlgenn import plotting

ROW_AS_SMB = bool(1)

_colors = ['red', 'green', 'blue', 'cyan', 'orange',
           'magenta', 'black', 'yellow', ]


def plot_images(order, shapes, test_y, kernels, spikes, sim_time,
                digit_duration, offsets, norm_w, n_digits, prefix):
    rates = {}
    conf_matrix = np.zeros((10, 10))
    correct = 0
    no_spikes = 0
    ncols = 5
    max_t = sim_time
    for si, k in enumerate(order):
        if k not in spikes:
            continue

        if k == 'dense':
            imgs, bins = plotting.spikes_to_images_list(
                                      spikes[k], shapes[k], max_t,
                                      digit_duration, offsets[k],
                                      merge_images=True)
            # rates[k] = bins
            rates[k] = [[np.mean([len(ts) for ts in b])
                        for b in pbins] for pbins in bins]

            nimgs = len(imgs)
            nrows = nimgs // ncols + int(nimgs % ncols > 0)

            fig = plt.figure(figsize=(ncols, nrows))
            plt.suptitle("{}_{}".format(k, 0))
            for i in range(nimgs):
                ax = plt.subplot(nrows, ncols, i + 1)
                ax.imshow(imgs[i])
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig("{}_{:03d}_{}_{:03d}.png".format(prefix, si, k, 0), dpi=150)
            plt.close(fig)
            continue

        rl = []
        for pi, p in enumerate(spikes[k]):
            imgs, bins = plotting.spikes_to_images(p, shapes[k], max_t,
                                                   digit_duration)
            rl.append([np.mean([len(ts) for ts in b]) for b in bins])

            nimgs = len(imgs)
            # if 'conv2d' in k:
            #     nimgs += 1
            nimgs += 1

            nrows = nimgs // ncols + int(nimgs % ncols > 0)

            fig = plt.figure(figsize=(ncols, nrows+1))
            plt.suptitle("{}_{}\n.".format(k, pi))
            for i in range(nimgs):
                if i == len(imgs):
                    break

                ax = plt.subplot(nrows, ncols, i + 1)
                if k == 'dense_2':
                    if np.sum(imgs[i]) == 0:
                        no_spikes += 1
                    else:
                        pred = int(np.argmax(imgs[i]))
                        corr = int(test_y[i])
                        
                        if pred == corr:
                            correct += 1
                        conf_matrix[corr, pred] += 1
                        ax.set_title("e {} - p {}".format(test_y[i], np.argmax(imgs[i])))

                ax.imshow(imgs[i])
                ax.set_xticks([])
                ax.set_yticks([])

            if 'conv2d' in k or 'dense' in k:
                w = kernels[k][pi]

                # if 'conv2d' in k:
                #     w = np.fliplr(np.flipud(w))
                #     # w = norm_w(w)

                vmax = np.max(np.abs(w))
                ax = plt.subplot(nrows, ncols, nimgs)
                im = ax.imshow(w, vmin=-vmax, vmax=vmax, cmap='PiYG')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im)

            plt.savefig("{}_{:03d}_{}_{:03d}.png".format(prefix, si, k, pi), dpi=150)
            plt.close(fig)

        rates[k] = rl

    return rates, conf_matrix, correct, no_spikes


def plot_matrix(conf_matrix, n_digits, no_spikes, correct, prefix):
    fig = plt.figure()
    plt.suptitle("confusion matrix ({})\n no spikes {} acc {:5.2f}%".format(
                    n_digits, no_spikes, (100. * correct)/n_digits))
    im = plt.imshow(conf_matrix)
    plt.colorbar(im)
    plt.savefig("{}_confusion_matrix.png".format(prefix), dpi=150)
    plt.close(fig)
    # plt.show()


def plot_rates(rates, order, colors=_colors, prefix=''):
    for i, k in enumerate(rates):
        if k == 'input':
            continue
        # max_r = np.max(rates[k])
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        for j, r in enumerate(rates[k]):

            lbl = "{} {}".format(k, j)
            plt.plot(r[:-1], label=lbl, linewidth=1)

        plt.plot(rates['input'][0][:-1], linestyle=':', color=colors[0],
                 label='Input', linewidth=4)


        # plt.legend()
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.savefig('{}_average_rates_simple_cnn_mnist_layer_{}.png'.format(
                    prefix, k), dpi=150)

        plt.close(fig)

        ord = order
        mean_rates = [np.mean(rates[rk]) for rk in ord if rk in rates]
        std_rates = [np.std(rates[rk]) for rk in ord if rk in rates]
        labels = [k for k in ord if k in rates]
        xticks = np.arange(len(mean_rates))
        fig, ax = plt.subplots(1, 1)
        ax.set_title("average rates per layer")
        im = plt.errorbar(xticks, mean_rates, yerr=std_rates,
                          linestyle='dotted', marker='o')
        ax.set_ylabel("rate (hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # ax.set_xticklabels(labels)
        plt.savefig("{}_average_spikes_per_layer.png".format(prefix), dpi=300)
        # plt.show()


def plot_spikes(order, spikes, sim_time, digit_duration, prefix):
    for si, k in enumerate(order):
        if k not in spikes:
            continue

        for pi, p in enumerate(spikes[k]):
            fig = plt.figure()
            plt.suptitle("{}_{}".format(k, pi))
            ax = plt.subplot(1, 1, 1)
            if k == 'dense_2':
                for i in range(10):
                    plt.axhline(i, linestyle='--', color='gray', linewidth=0.1)

            for i in np.arange(0, sim_time, digit_duration):
                plt.axvline(i, linestyle='--', color='gray', linewidth=0.5)

            for ni, ts in enumerate(p):
                ax.plot(ts, ni * np.ones_like(ts), '.b', markersize=1)
            plt.savefig("{}_spikes_{:03d}_{}_{:03d}.png".format(
                prefix, si, k, pi), dpi=300)
            plt.close(fig)

# plt.show()


