import numpy as np
import matplotlib.pyplot as plt

dumps_per_rate = {
    5: [0, 0, 0, 0, 0, 0, 4, 6],
    10: [0, 0, 0, 0, 0, 0, 9, 37],
    25: [0, 0, 1, 181, 1204, 3426, 6153, 8864],
    50: [0, 68, 2174, 7877, 13170, 18043, 22405, 26850],
    100: [0, 5580, 16308, 25460, 33980, 42078, 53812, 61430],
    150: [111, 14327, 28150, 40500, 56905, 64934, 71760, 78475],
    200: [1604, 22257, 38695, 58896, 68124, 77028, 85174, 94407],
}


x_ticks = sorted(dumps_per_rate.keys())
dumps_per_input = {i+1: [] for i in range(len(dumps_per_rate[5]))}
for k in x_ticks:
    for i, v in enumerate(dumps_per_rate[k]):
        dumps_per_input[i+1].append(v)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.axhspan(38000, 56000, color='orange', label="Back pressure", alpha=0.25)
plt.axhspan(56000, 95000, color='red', label="Dead", alpha=0.25)
for i in dumps_per_input:
    ax.semilogy(x_ticks, dumps_per_input[i], label="{} sources".format(i),
                marker='.', linewidth=2)
ax.set_xticks(x_ticks)
ax.set_xlabel("Input rate (Hz)")
ax.set_ylabel("Dumped spikes")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig("dumped_spikes_per_input_rate_and_num_sources.pdf")
plt.show()


