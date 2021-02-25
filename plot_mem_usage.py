import matplotlib.pyplot as plt
import numpy as np

# data from experiments
# https://docs.google.com/spreadsheets/d/1ZqDyE_dGI4wBILl6NPwy97T21w4VF9g9vWWcL5GuOho/edit?usp=sharing
x = [1, 1, 4, 25, 64, 121, 169, 256, 361, 400, 484, 625, 729, 900, 1024]
rep = [0.7421875, 0.7890625, 1, 2, 3, 6, 8, 11, 15, 24, 25, 28, 30, 41, 44]
sark = [11, 11, 11, 12, 14, 16, 21, 23, 32, 33, 34, 52, 54, 57, 59]
vars = [
    0.01171875, 0.01171875, 0.046875, 0.29296875, 0.75, 1.41796875,
    1.98046875, 3, 4.23046875, 4.6875, 5.671875, 7.32421875, 8.54296875,
    10.546875, 12
]

mrep, brep = np.polyfit(x, rep, 1)
msark, bsark = np.polyfit(x, sark, 1)
mvars, bvars = np.polyfit(x, vars, 1)

x_lag = 675
x_behind = 729
x_no_mem = 1089
pre_splits = [400, 900]

y_nm_rep = mrep*x_no_mem + brep
y_nm_sark = msark*x_no_mem + bsark
y_nm_vars = mvars*x_no_mem + bvars

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(x, rep, color='cyan', label='Report')
ax.plot(x, sark, color='blue', label='Sark')
ax.plot(x, vars, color='green', label='State Vars')

ax.plot([x[-1], x_no_mem], [rep[-1], y_nm_rep], ':', color='cyan')
ax.plot([x[-1], x_no_mem], [sark[-1], y_nm_sark], ':', color='blue')
ax.plot([x[-1], x_no_mem], [vars[-1], y_nm_vars], ':', color='green')

plt.axvspan(x[-1], x_no_mem, color='gray', alpha=0.1, label='Projected',
            linewidth=0)
plt.axvline(x_lag, linestyle='--', color='orange', label='Drops packets')
plt.axvline(x_behind, linestyle='--', color='magenta', label='Lags')
plt.axvline(x_no_mem, linestyle='--', color='red', label='Breaks')
for i, sp in enumerate(pre_splits):
    plt.axvline(sp, linestyle='--', color='gray', alpha=0.5,
                label='Pre slices {}'.format(i+2))
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Memory usage per core [KB]')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('memory_usage_still_using_one_connector_per_pre_slice.pdf')
plt.show()