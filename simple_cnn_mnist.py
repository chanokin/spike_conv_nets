import spynnaker8 as sim
import numpy as np

filename = "simple_cnn_network_elements.npz"

data = np.load(filename, allow_pickle=True)

order = data['order']
ml_conns = data['conns'].item()
ml_param = data['params'].item()

print(data.keys())


