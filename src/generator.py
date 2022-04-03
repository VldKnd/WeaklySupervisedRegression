import torch
import torch.nn as nn


def he_constant(m):
    return torch.sqrt(torch.tensor(2 / m.weight[0].numel()))


class ConstrainedLayer(nn.Module):

    def __init__(self, module):
        super().__init__()
        module.bias.data.fill_(0)
        module.weight.data.normal_(0, 1)
        self._module = module
        self._weight = he_constant(self._module)

    def forward(self, x):
        return self._module(x) * self._weight


class EqualizedLinear(ConstrainedLayer):

    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__(
            nn.Linear(in_ch, out_ch, bias=bias),
        )


class Generator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self._first_layer = nn.Sequential(
            EqualizedLinear(in_ch=cfg['z_dim'], out_ch=cfg['gen_dims'][0]),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
        )
        self._layers = nn.Sequential(*[
            nn.Sequential(
                EqualizedLinear(in_ch=cfg['gen_dims'][i], out_ch=cfg['gen_dims'][i + 1]),
                nn.LeakyReLU(negative_slope=.2, inplace=True),
            )
            for i in range(len(cfg['gen_dims']) - 1)
        ])

    def forward(self, z):
        return self._layers(self._first_layer(z))
