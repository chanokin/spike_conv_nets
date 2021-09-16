#import tensorflow as tf
from bifrost.ir import (InputLayer, OutputLayer, DummyTestInputSource,
                        EthernetOutput, PoissonImageDataset)
from tensorflow.keras import models#, layers, datasets
from ml_genn import Model, save_model, load_model
# from ml_genn.layers import InputType
# from ml_genn.norm import DataNorm, SpikeNorm
# from ml_genn.utils import parse_arguments, raster_plot
import numpy as np

from bifrost.extract.ml_genn.extractor import extract_all
from bifrost.parse.parse_ml_genn import (to_neuron_layer, to_connection, ml_genn_to_network)
from bifrost.export.ml_genn import MLGeNNContext
from bifrost.export.population import export_layer_neuron
from bifrost.export.connection import export_connection
from bifrost.exporter import export_network
from bifrost.main import get_parser_and_saver, set_recordings

def to_dict(np_file):
    d = {}
    for k in np_file.keys():
        try:
            d[k] = np_file[k].item()
        except:
            d[k] = np_file[k]
    return d


tf_model = models.load_model('simple_cnn_tf_model.tf')
mlg_model = Model.convert_tf_model(tf_model, input_type='poisson', 
                connectivity_type='procedural')
mlg_model.compile(dt=1.0, batch_size=1, rng_seed=0)

my_model = mlg_model

# import dill
# with open('simple_cnn_mlgenn_model.mlg', 'wb') as f:
#     dill.dump(mlg_model, f)
#
# with open('simple_cnn_mlgenn_model.mlg', 'rb') as f:
#     my_model = dill.load(f)
#     print(my_model.__class__.__name__)

# save_model(mlg_model, 'simple_cnn_tf_model.mlg')
# my_model = load_model('simple_cnn_tf_model.mlg')


tab = " " * 4
parser, saver = get_parser_and_saver(my_model)
in_shape = [28, 28]
in_size = int(np.prod(in_shape))
n_samples = 5
load_mnist_images = "\n".join([
    # f'def _load_mnist(start_sample, num_samples, num_channels):',
    f'{tab}X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)',
    f'{tab}X = X.reshape((X.shape[0], -1))',
    f'{tab}X_train, X_test, y_train, y_test = train_test_split(',
    f'{tab * 2}X, y, train_size=5000,',
    f'{tab * 2}test_size=10000, shuffle=False)',
    f'{tab}return ({{0: X_test[start_sample: start_sample + num_samples]}},',
    f'{tab * 3}y_test[start_sample: start_sample + num_samples])\n ',
])

transform_to_rates = "\n".join([
    # def transform(images_dictionary):
    f"{tab}return {{k: (1.0 / 255.0) * images_dictionary[k] \n"
    f"{tab * 3}for k in images_dictionary}}"
])

imports = [
    'from sklearn.datasets import fetch_openml',
    'from sklearn.model_selection import train_test_split',
    # 'from sklearn.utils import check_random_state'
]
source = PoissonImageDataset(
    defines={},
    imports=imports,
    shape=in_shape,
    load_command_body=load_mnist_images,
    start_sample=0,
    num_samples=n_samples,
    pixel_to_rate_transform=transform_to_rates,
    on_time_ms=400.0,
    off_time_ms=100.0,
)
# inp = InputLayer("in", 768, 1, DummyTestInputSource([28, 28]))
inp = InputLayer("in", in_size, 1, source=source)
out = None  # OutputLayer("out", 1, 1, sink=EthernetOutput())

record = {'spikes': [0, 1, -1], 'v': [1]}
net, context, net_params = ml_genn_to_network(my_model, inp, out,
                                              config={'runtime': 10.0})
set_recordings(net, record)
np.savez_compressed('ml_genn_as_spynn_net_dict.npz', **net_params)
# result = export_network(net, context)
with open("ml_genn_as_spynn_net.py", 'w') as f:
    f.write(export_network(net, context,))
# print(result)

