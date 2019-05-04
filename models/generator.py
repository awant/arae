import torch
from torch import nn
from collections import OrderedDict


class Gen(nn.Module):
    # The same as Discriminator, except that has different activation func
    def __init__(self, ninput, noutput, layer_sizes):
        super().__init__()
        layers = []

        self.inp_size = inp_size = ninput
        for idx, layer_size in enumerate(layer_sizes):
            layer = nn.Linear(inp_size, layer_size)
            layers.append(('linear_{}'.format(idx), layer))

            if idx > 0:
                bn = nn.BatchNorm1d(layer_size, eps=1e-05, momentum=0.1)
                layers.append(('bn_{}'.format(idx), bn))

            layers.append(('activation_{}'.format(idx), nn.ReLU()))
            inp_size = layer_size

        layer = nn.Linear(inp_size, noutput)
        layers.append(('linear_last', layer))

        self.layer = nn.Sequential(OrderedDict(layers))

        self.init_weights()

    @classmethod
    def from_opts(cls, opts):
        layer_sizes = list(map(int, opts.arch_g.split('-')))
        return cls(opts.z_size, opts.nhidden, layer_sizes)

    def init_weights(self):
        init_std = 0.02
        for layer in self.layer:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def forward(self, inp):
        return self.layer(inp)
