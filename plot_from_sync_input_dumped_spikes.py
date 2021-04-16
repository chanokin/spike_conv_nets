import numpy as np
import matplotlib.pyplot as plt

dumps_per_rate = {
    1: [0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0],
    10: [0, 0, 0, 0, 0],
    15: [1, 2, 3, 11, 14],
    20: [2, 8, 17, 45, 58],
    40: [15, 33, 57, 112, 152],
    70: [36, 81, 128, 260, 352],
    100: [70, 143, 210, 424, 558],
    140: [92, 203, 312, 623, 854],
    180: [132, 262, 423, 821, 1108],
    225: [154, 313, 479, 979, 1323],
}
start_of_back_pressure = np.asarray([
    [70, 81],
    [40, 57],
    [40, 112],
    [40, 152]
])

labels = [1, 2, 3, 6, 8]
x_ticks = sorted(dumps_per_rate.keys())
dumps_per_input = {labels[i]: [] for i in range(len(dumps_per_rate[5]))}
for k in x_ticks:
    for i, v in enumerate(dumps_per_rate[k]):
        dumps_per_input[labels[i]].append(v)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# plt.axhspan(38000, 56000, color='orange', label="Back pressure", alpha=0.25)
# plt.axhspan(56000, 95000, color='red', label="Dead", alpha=0.25)
for i in dumps_per_input:
    ax.semilogy(x_ticks, dumps_per_input[i], label="{} sources".format(i),
                marker='.', linewidth=2)

ax.plot(start_of_back_pressure[:, 0],
        start_of_back_pressure[:, 1], 'x', color='black', markersize=8,
        label='Back pressure')

ax.set_xticks(x_ticks)
ax.set_xlabel("Number of simultaneous input spikes")
ax.set_ylabel("Dumped spikes")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig("sync_input_dumped_spikes_per_input_rate_and_num_sources.pdf")
plt.show()


