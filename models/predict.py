"""
Classes:

- FeedforwardNeuralNetModel: Base class for the feed-forward neural network model.
- CNNModel: Base class for the CNN-based neural network model.

Methods:

- predict_nn: Method to predict added mass and potential damping tensors using the feed-forward NN models.
- predict_cnn: Method to predict added mass and potential damping tensors using the CNN-based models.
- image_from_polygon: Method to generate a binary image suited as input for the CNN-based models from a list of vertices.

"""

__author__ = 'Marius Seidl'
__date__ = '2025-05-16'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'


# standard library imports
import numpy as np
import torch
import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 36)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def predict_nn(vertices: np.ndarray, thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict added mass tensors and potential damping tensor from floe shape and thickness.

    :param vertices: Vertices of polygon making up floe polygonal prism shape as numpy array.
    :param thickness: Thickness of floe as float.
    :return: Tuple of added mass tensors (0 and infinite wave frequency) and potential damping tensor as numpy arrays.
    """
    # validate inputs
    if len(vertices.shape) != 2:
        raise ValueError('vertices must be a 2D array')
    if vertices.shape[1] != 2:
        raise ValueError('vertices must have 2 columns')
    if vertices.shape[0] < 3 or vertices.shape[1] > 12:
        raise ValueError('vertices must have between 3 and 12 rows (other ML models are unavailable)')

    # load models
    added_mass_0 = FeedforwardNeuralNetModel(vertices.shape[0]*2+1, 1024)
    added_mass_0.load_state_dict(
        torch.load('./models/model_added_mass_0_wave_frequency_n{}.pth'.format(vertices.shape[0]),
                   map_location=torch.device('cpu'),
                   weights_only=True))
    added_mass_inf = FeedforwardNeuralNetModel(vertices.shape[0]*2+1, 8192)
    added_mass_inf.load_state_dict(
        torch.load('./models/model_added_mass_infinite_wave_frequency_n{}.pth'.format(vertices.shape[0]),
                   map_location=torch.device('cpu'),
                   weights_only=True))
    potential_damping = FeedforwardNeuralNetModel(vertices.shape[0]*2+1, 256)
    potential_damping.load_state_dict(
        torch.load('./models/model_potential_damping_n{}.pth'.format(vertices.shape[0]),
                   map_location=torch.device('cpu'),
                   weights_only=True))

    # set models to evaluation mode
    added_mass_0.eval()
    added_mass_inf.eval()
    potential_damping.eval()

    # make prediction
    inputs = torch.from_numpy(np.hstack((vertices.flatten(), thickness))).to(dtype=torch.float32)
    with torch.no_grad():
        added_mass_0_tensor = added_mass_0(inputs).numpy()
        added_mass_inf_tensor = added_mass_inf(inputs).numpy()
        potential_damping_tensor = potential_damping(inputs).numpy()

    # reverse normalization
    added_mass_0_tensor = (added_mass_0_tensor
                           * np.load('./models/added_mass_0_wave_frequency_norm_mult.npy')
                           + np.load('./models/added_mass_0_wave_frequency_norm_add.npy')).reshape((6,6))
    added_mass_inf_tensor = (added_mass_inf_tensor
                           * np.load('./models/added_mass_infinite_wave_frequency_norm_mult.npy')
                           + np.load('./models/added_mass_infinite_wave_frequency_norm_add.npy')).reshape((6,6))
    potential_damping_tensor = (potential_damping_tensor
                           * np.load('./models/potential_damping_norm_mult.npy')
                           + np.load('./models/potential_damping_norm_add.npy')).reshape((6,6))

    return added_mass_0_tensor, added_mass_inf_tensor, potential_damping_tensor


class CNNModel(nn.Module):
    def __init__(self, hidden_dim):
        super(CNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.convolution_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, stride=4, padding=5), #32x32x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #16x16x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2), #16x16x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #8x8x32
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2050, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 36),
        )

    def forward(self, x):
        image, inputs = x
        image = image[None, None, :, :]
        conv_out = self.convolution_relu_stack(image)
        lin_in = torch.hstack((self.flatten(conv_out)[0], inputs))
        out = self.linear_relu_stack(lin_in)
        return out


def predict_cnn(image: np.ndarray, pixel_size: float, thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict added mass tensors and potential damping tensor from 128x128 binary image of floe shape, image pixel size
    and thickness.

    :param image: 128x128 binary image of floe shape as numpy array.
    :param pixel_size: Pixel size of floe image as float.
    :param thickness: Thickness of floe as float.
    :return: Tuple of added mass tensors (0 and infinite wave frequency) and potential damping tensor as numpy arrays.
    """
    # validate inputs
    if len(image.shape) != 2 or image.shape[0] != 128 or image.shape[1] != 128:
        raise ValueError('image must be a 128x128 array')

    # load models
    added_mass_0 = CNNModel(512)
    added_mass_0.load_state_dict(
        torch.load('./models/model_added_mass_0_wave_frequency_cnn.pth',
                   map_location=torch.device('cpu'),
                   weights_only=True))
    added_mass_inf = CNNModel(8192)
    added_mass_inf.load_state_dict(
        torch.load('./models/model_added_mass_infinite_wave_frequency_cnn.pth',
                   map_location=torch.device('cpu'),
                   weights_only=True))
    potential_damping = CNNModel(256)
    potential_damping.load_state_dict(
        torch.load('./models/model_potential_damping_cnn.pth',
                   map_location=torch.device('cpu'),
                   weights_only=True))

    # set models to evaluation mode
    added_mass_0.eval()
    added_mass_inf.eval()
    potential_damping.eval()

    # make prediction
    inputs = (torch.from_numpy(image).to(dtype=torch.float32),
              torch.tensor((pixel_size, thickness)).to(dtype=torch.float32),)
    with torch.no_grad():
        added_mass_0_tensor = added_mass_0(inputs).numpy()
        added_mass_inf_tensor = added_mass_inf(inputs).numpy()
        potential_damping_tensor = potential_damping(inputs).numpy()

    # reverse normalization
    added_mass_0_tensor = (added_mass_0_tensor
                           * np.load('./models/added_mass_0_wave_frequency_norm_mult.npy')
                           + np.load('./models/added_mass_0_wave_frequency_norm_add.npy')).reshape((6, 6))
    added_mass_inf_tensor = (added_mass_inf_tensor
                             * np.load('./models/added_mass_infinite_wave_frequency_norm_mult.npy')
                             + np.load('./models/added_mass_infinite_wave_frequency_norm_add.npy')).reshape((6, 6))
    potential_damping_tensor = (potential_damping_tensor
                                * np.load('./models/potential_damping_norm_mult.npy')
                                + np.load('./models/potential_damping_norm_add.npy')).reshape((6, 6))

    return added_mass_0_tensor, added_mass_inf_tensor, potential_damping_tensor


def image_from_polygon(vertices: np.ndarray, resolution: int = 128):
    """
    Create a binary image from the vertices of a polygon as valid input for the CNN model.

    :param vertices: Two-column array of vertices.
    :param resolution: Number of pixels in one dimension.
    :return: Tuple of image array, pixel size.
    """
    # prepare data
    x_lb, x_ub = np.min(vertices[:,0]), np.max(vertices[:,0])
    y_lb, y_ub = np.min(vertices[:,1]), np.max(vertices[:,1])
    lb = min(x_lb, y_lb)
    lb -= 0.05 * abs(lb)
    ub = max(x_ub, y_ub)
    ub += 0.05 * abs(ub)
    pixel_size = (ub-lb)/resolution
    x_range = np.linspace(lb, ub, resolution)
    y_range = np.linspace(lb, ub, resolution)

    # make pixel centre grid
    x, y = np.meshgrid(x_range, y_range)

    # check if pixel centre is inside polygon by counting intersections of ray with polygon boundary
    n_intersection = np.zeros(x.shape, dtype=np.int64)
    for (x1, y1), (x2, y2) in zip(vertices, np.vstack((vertices[1:,:], vertices[0,:]))):
        between = np.logical_or(np.logical_and(y >= y1, y <= y2), np.logical_and(y <= y1, y >= y2))
        between[np.logical_or(np.logical_and(y == y1, y2 >= y1), np.logical_and(y == y2, y1 >= y2))] = False
        cross_product = (x1-x) * (y2-y) - (x2-x) * (y1-y)
        n_intersection += np.logical_and((y1 < y2) == (cross_product > 0), between)

    # odd number of intersections -> inside polygon (do bit check for speed)
    return (n_intersection & 0x1).astype(np.float32), pixel_size
