import numpy as np
import field_encoding as fe
import matplotlib.pyplot as plt


def set_fontsize(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)


max_w = 512
max_h = 512
widths = np.arange(1, max_w, dtype='int')
heights = np.arange(1, max_h, dtype='int')
ROWS_AS_MSB = bool(1)

pot_sizes = np.zeros((max_h, max_w))
coord_sizes = np.zeros((max_h, max_w))

for w in widths:
    for h in heights:
        pot_sizes[h, w] = fe.power_of_2_size(w, h, ROWS_AS_MSB)
        coord_sizes[h, w] = fe.max_coord_size(w, h, ROWS_AS_MSB)

vmin = min(pot_sizes.min(), coord_sizes.min())
vmax = max(pot_sizes.max(), coord_sizes.max())

fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
im = ax0.imshow(pot_sizes, vmin=vmin, vmax=vmax)
ax0.set_xlabel("width")
ax0.set_ylabel("height")
ax0.invert_yaxis()
set_fontsize(ax0)

im = ax1.imshow(coord_sizes, vmin=vmin, vmax=vmax)
ax1.set_xlabel("width")
ax1.invert_yaxis()
set_fontsize(ax1)

cax = fig.add_axes([0.915, 0.11, 0.02, 0.77])
fig.colorbar(im, cax=cax)
set_fontsize(cax)
# plt.tight_layout()
plt.show()

