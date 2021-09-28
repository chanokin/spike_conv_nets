import torch
import numpy as np
from mnist_norse import LIFConvNet as NorseModel
from bifrost.parse.parse_torch import torch_to_network, torch_to_context
from bifrost.ir import (InputLayer, PoissonImageDataset)
from bifrost.main import get_parser_and_saver


model = NorseModel(28*28, 1)

checkpoint = torch.load('mnist-final.pt',
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

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
    f'{tab}X, y, train_size=5000,',
    f'{tab}test_size=10000, shuffle=False)',
    f'X_test = (X_test[start_sample: start_sample + num_samples].T)',
    f'shape_in = np.asarray({in_shape})',
    f'n_in = int(np.prod(shape_in))',
    f'in_ids = np.arange(0, n_in)',
    f'xy_in_ids = fe.convert_ids(in_ids, shape=shape_in)',
    f'small = np.where(xy_in_ids < np.prod(shape_in))',
    # in_ids = in_ids[small]
    f'xy_in_ids = xy_in_ids[small]',
    f'for i in range(num_samples):',
    f'{tab}X_test[xy_in_ids, i] = X_test[small, i]',
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
    'import field_encoding as fe'
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
bf_input = InputLayer("in", in_size, 1, source=source)
bf_net = parser(model, bf_input)

print(bf_net)
