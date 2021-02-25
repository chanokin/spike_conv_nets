import matplotlib.pyplot as plt
import numpy as np

# data from experiments
# https://docs.google.com/spreadsheets/d/1ZqDyE_dGI4wBILl6NPwy97T21w4VF9g9vWWcL5GuOho/edit?usp=sharing
x = [1, 1, 1, 1, 9, 16, 25, 49, 64, 100, 100, 121, 144, 196, 225, 289, 324,
     361, 441, 484, 576, 625, 676]
rep = [0.79296875, 0.83984375, 1, 1, 2, 4, 5, 7, 9, 18, 18, 18, 19, 28, 28, 37,
       38, 39, 48, 49, 59, 68, 69]
sark = [11, 11, 11, 11, 11, 11, 12, 13, 14, 16, 16, 17, 21, 22, 22, 31, 32, 32,
        34, 34, 52, 53, 54]
vars = [0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.10546875, 0.1875,
        0.29296875, 0.57421875, 0.75, 1.171875, 1.171875, 1.41796875, 1.6875,
        2.296875, 2.63671875, 3.38671875, 3.796875, 4.23046875, 5.16796875,
        5.671875, 6.75, 7.32421875, 7.921875
]

mrep, brep = np.polyfit(x, rep, 1)
msark, bsark = np.polyfit(x, sark, 1)
mvars, bvars = np.polyfit(x, vars, 1)

x_lag = 144
x_behind = 576
# x_no_mem = 1089
pre_splits = [100, 196, 289, 441, 484, 625]

# y_nm_rep = mrep*x_no_mem + brep
# y_nm_sark = msark*x_no_mem + bsark
# y_nm_vars = mvars*x_no_mem + bvars

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(x, rep, color='cyan', label='Report')
ax.plot(x, sark, color='blue', label='Sark')
ax.plot(x, vars, color='green', label='State Vars')

# ax.plot([x[-1], x_no_mem], [rep[-1], y_nm_rep], ':', color='cyan')
# ax.plot([x[-1], x_no_mem], [sark[-1], y_nm_sark], ':', color='blue')
# ax.plot([x[-1], x_no_mem], [vars[-1], y_nm_vars], ':', color='green')

# plt.axvspan(x[-1], x_no_mem, color='gray', alpha=0.1, label='Projected',
#             linewidth=0)
plt.axvline(x_lag, linestyle='--', color='orange', label='Drops packets')
plt.axvline(x_behind, linestyle='--', color='magenta', label='Lags')
# plt.axvline(x_no_mem, linestyle='--', color='red', label='Breaks')
for i, sp in enumerate(pre_splits):
    plt.axvline(sp, linestyle='--', color='gray', alpha=0.5,
                label='Pre slices {}'.format(i+2))
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Memory usage per core [KB]')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('pool_memory_usage_still_using_one_connector_per_pre_slice.pdf')
plt.show()