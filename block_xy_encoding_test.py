import numpy as np
import matplotlib.pyplot as plt
from spynnaker.pyNN.utilities.neuron_id_encoding import BlockXYEncoder

block_width = 8
block_height = 8
dvs_width = 24
dvs_height = 16

bxy_encoder = BlockXYEncoder(shape=(dvs_height, dvs_width),
                             block_shape=(block_height, block_width))

n_vertical, n_horizontal = bxy_encoder.n_blocks

sq_w = 0.75
fig = plt.figure(figsize=(sq_w * dvs_width, sq_w * dvs_height))
fig.patch.set_facecolor('white')

ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal', adjustable='box')
ax.set_ylim(dvs_height, 0)
ax.axis('off')

for row in range(dvs_height + 1):
    ax.plot([0, dvs_width], [row, row], linewidth=0.5, color='gray')

for col in range(dvs_width + 1):
    ax.plot([col, col], [0, dvs_height], linewidth=0.5, color='gray')

for br in range(n_vertical + 1):
    row = br * block_height
    ax.plot([0, dvs_width], [row, row], 'r', linewidth=4)

for bc in range(n_horizontal + 1):
    col = bc * block_width
    ax.plot([col, col], [0, dvs_height], 'r', linewidth=4)

for row in range(dvs_height):
    for col in range(dvs_width):
        e = bxy_encoder.encode_coords(row, col)
        ax.text(col, row + 0.6, "{: 3d}".format(e),
                fontsize=13, fontweight='bold', fontfamily='monospace')


plt.show()
