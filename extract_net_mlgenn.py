#import tensorflow as tf
from bifrost.ir import InputLayer, OutputLayer, InputSource, EthernetOutput
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


inp = InputLayer("in", 768, 1, InputSource([28, 28]))
out = None  # OutputLayer("out", 1, 1, sink=EthernetOutput())

net, context, net_params = ml_genn_to_network(my_model, inp, out)
result = export_network(net, context)

print(result)

