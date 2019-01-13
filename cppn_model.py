import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import itertools

_scale = 0.1


class SinActivation(nn.Module):
    """
    nn.Module wrapper for torch.sin
    """

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CosActivation(nn.Module):
    """
    nn.Module wrapper for torch.cos
    """

    def __init__(self):
        super(CosActivation, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class TruncatedGaussActivation(nn.Module):
    """
    nn.Module applying the Gauss function
    """

    def __init__(self):
        super(TruncatedGaussActivation, self).__init__()
        self.mean = 0
        self.std = 0.4
        self.min = -1
        self.max = 1

    def forward(self, x):
        gauss = torch.exp((-(x - self.mean) ** 2) / (2 * self.std ** 2))
        return torch.clamp(gauss, min=self.min, max=self.max)


ACTIVATIONS = [nn.Tanh(),
               SinActivation(),
               CosActivation()
               ]


class CPPN(nn.Module):
    """
    A pytorch module defining a CPPN. Supports dynamic net depth and mutable weights.
    """
    _scale = 0.1

    # net_depths = [2, 4, 6, 8, 10, 12, 14, 16]
    # net_sizes = [8, 12, 16, 24, 32]
    # net_depths = [12]
    # net_sizes = [32]
    net_depths = [3, 4, 5, 6, 8, 12, 16, 20, 24]
    # net_depths = [8, 12, 16, 20, 24]
    net_sizes = [32]
    combinations = list(itertools.product(net_depths, net_sizes))

    def __init__(self, color=True, net_depth=None, net_size=None, input_size=11, img_size=64, z=None):
        super(CPPN, self).__init__()
        self.param_idx = 0

        self.net_depth = net_depth
        self.net_size = net_size

        self.img_size = img_size
        if input_size < 5:
            raise ValueError('input_size must be >5')
        self.input_size = input_size
        if z is not None:
            self.z = z
        else:
            self.z = torch.rand(self.input_size - 3)
        self.color = color
        self.input_tensor = self.__create_cppn_input()

        self.initialize_net()

    def initialize_net(self):
        if not self.net_size:
            self.net_size = CPPN.combinations[self.param_idx][1]
        if not self.net_depth:
            self.net_depth = CPPN.combinations[self.param_idx][0]
        self.param_idx += 1
        self.param_idx %= len(CPPN.combinations)
        self.z = torch.rand(self.input_size - 3)

        # Define net
        self.net = nn.Sequential()
        self.net.add_module('layer1', nn.Linear(self.input_size, self.net_size, bias=False))
        self.net.add_module('act1', nn.Tanh())

        for i in range(2, self.net_depth - 1):
            self.net.add_module(f'layer{i}', nn.Linear(self.net_size, self.net_size, bias=False))
            self.net.add_module(f'act{i}', ACTIVATIONS[0])  # nn.Tanh())

        self.net.add_module(f'layer{self.net_depth - 1}', nn.Linear(self.net_size, self.net_size, bias=False))
        self.net.add_module(f'act{self.net_depth - 1}', ACTIVATIONS[0])

        if self.color:
            self.net.add_module(f'layer{self.net_depth}', nn.Linear(self.net_size, 3, bias=False))
        else:
            self.net.add_module(f'layer{self.net_depth}', nn.Linear(self.net_size, 1, bias=False))

        self.net.add_module(f'act{self.net_depth}', nn.Sigmoid())

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1, b=1)

    def reset(self, reset_depth=True, reset_size=True):
        if reset_depth:
            self.net_depth = None
        if reset_size:
            self.net_size = None
        self.initialize_net()

    def forward(self, x):
        return self.net(x)

    def render_image(self):
        image = self(self.input_tensor).data.numpy()
        if self.color:
            image = image.reshape(self.img_size, self.img_size, 3)
        else:
            image = image.reshape(self.img_size, self.img_size)
        return image

    def get_weight_shapes(self):
        return [W.detach().numpy().shape for W in list(self.parameters())]

    def get_flat_weight_shapes(self):
        shapes = self.get_weight_shapes()
        return [np.prod(s) for s in shapes]

    def get_weights(self):
        weights = list(self.parameters())
        weights = [W.detach().numpy() for W in weights]
        shapes = [W.shape for W in weights]
        weights = [W.flatten() for W in weights]
        flat_shapes = [W.shape for W in weights]
        weights = np.concatenate(weights)
        return weights

    def set_weights(self, weights: np.array):
        flat_shapes = self.get_flat_weight_shapes()
        if not weights.size == sum(flat_shapes):
            raise ValueError('cant reshape weights from flat to layer matrices')

        split_indices = [flat_shapes[i] + sum(flat_shapes[:i]) for i in range(len(flat_shapes))]
        weights = np.split(weights, split_indices)
        shapes = self.get_weight_shapes()
        weights = [np.reshape(W, s) for W, s in zip(weights, shapes)]
        i = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.Parameter(torch.from_numpy(weights[i]))
                i += 1

    def set_img_size(self, img_size):
        self.img_size = img_size
        self.input_tensor = self.__create_cppn_input()

    def __create_cppn_input(self):
        indices = torch.arange(0, self.img_size)
        grid = torch.stack(
            [indices.repeat(self.img_size, 1).t().contiguous().view(-1), indices.repeat(self.img_size)]).float()
        radiuses = torch.sqrt(
            ((grid[0] * _scale - (self.img_size * _scale / 2)) ** 2) + (
                    (grid[1] * _scale - (self.img_size * _scale / 2)) ** 2))
        grid = grid * _scale
        z_size = len(self.z)
        input_tensor = torch.zeros(1, z_size + 3).expand(self.img_size * self.img_size, z_size + 3)
        input_tensor = input_tensor + torch.cat((self.z, torch.zeros(3)))

        input_tensor[:, z_size] = grid[0]
        input_tensor[:, z_size + 1] = grid[1]
        input_tensor[:, z_size + 2] = radiuses
        input_tensor = Variable(input_tensor)
        return input_tensor
