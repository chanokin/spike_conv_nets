import torch
import numpy as np
from mnist_norse import LIFConvNet as NorseModel
from bifrost.parse.parse_torch import torch_to_network, torch_to_context
from bifrost.ir import (InputLayer, PoissonImageDataset)
from bifrost.main import get_parser_and_saver, set_recordings
from bifrost.exporter import export_network
from bifrost.export.configurations import SUPPORTED_CONFIGS
from bifrost.extract.torch.parameter_buffers import set_parameter_buffers


model = NorseModel(28*28, 1)
filename = 'mnist-final-100-poisson.pt'
checkpoint = torch.load(filename,
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=False)

# set_parameter_buffers(model)
# torch.save(
#     dict(state_dict=model.state_dict(keep_vars=True)),
#     filename,
# )
print(model)

tab = " " * 4
parser, saver = get_parser_and_saver(model)
in_shape = [28, 28]
in_size = int(np.prod(in_shape))
n_samples = 5
load_mnist_images = f"\n{tab}".join([
    # f'def _load_mnist(start_sample, num_samples, num_channels):',
    f'{tab}X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)',
    f'X = X.reshape((X.shape[0], -1))',
    f'X_train, X_test, y_train, y_test = train_test_split(',
    f'{tab}X, y, train_size=60000,',
    f'{tab}test_size=10000, shuffle=False)',
    f'X_test = (X_test[start_sample: start_sample + num_samples].T)',
    f'y_test = y_test[start_sample: start_sample + num_samples]',
    f'return ({{0: X_test}}, y_test)\n ',
])

transform_to_rates = f"\n{tab}".join([
    # def transform(images_dictionary):
    f"{tab}return {{k: (100.0 / 255.0) * images_dictionary[k] \n"
    f"{tab * 2}for k in images_dictionary}}"
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

config = {
    # 'runtime': 0.0,
    'split_runs': True,
    'configuration': {
        SUPPORTED_CONFIGS.MAX_NEURONS_PER_COMPUTE_UNIT: [('IF_curr_exp', (32, 16))],
        SUPPORTED_CONFIGS.MAX_NEURONS_PER_LAYER_TYPE: [
            ('dense', (128, 1)), ('conv2d', (32, 16)),
        ],
    },
}

# inp = InputLayer("in", 768, 1, DummyTestInputSource([28, 28]))
bf_input = InputLayer("in", in_size, 1, source=source)
bf_net, bf_context, bf_net_dict = parser(model, bf_input, config=config)

record = {
    'spikes': [0, 1, 2, 3, 4, 5],
    # 'spikes': [-1],
    'v': [-1]
}
set_recordings(bf_net, record)
# np.savez_compressed('ml_genn_as_spynn_net_dict.npz', **net_params)
# result = export_network(net, context)
with open("torch_as_spynn_net.py", 'w') as f:
    f.write(export_network(bf_net, bf_context,))

