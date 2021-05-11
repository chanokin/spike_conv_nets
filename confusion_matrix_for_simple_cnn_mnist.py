import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

with h5py.File("output_data_simple_cnn_mnist.h5", "r") as h5:
    correct = 0
    mtx = np.zeros((10, 10))
    smp = "sample"
    tgt = "target"
    rts = "rates"
    nt = "n_test"

    all_rates = h5[rts]
    targets = h5[tgt]
    n_tests = h5[nt][0]
    for idx in range(n_tests):
        rates = np.asarray(all_rates[idx])
        pred = np.argmax(rates)
        expect = int(targets[idx])

        if pred == expect:
            correct += 1

        mtx[pred, expect] += 1
        mtx[expect, pred] += 1

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(mtx)
    ax.set_title("accuracy {:6.2f}% {}/{}".format(
        100.*(correct / (idx + 1.)),
        correct, idx + 1))
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix_for_simple_cnn_mnist.pdf")
    plt.show()