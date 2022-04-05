import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self._first_layer = nn.Sequential(
            nn.Linear(in_features=cfg['z_dim'], out_features=cfg['gen_dims'][0]),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
        )
        self._layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=cfg['gen_dims'][i], out_features=cfg['gen_dims'][i + 1]),
                nn.LeakyReLU(negative_slope=.2, inplace=True) if i != len(cfg['gen_dims']) - 1 else nn.Identity(),
            )
            for i in range(len(cfg['gen_dims']) - 1)
        ])

    def forward(self, z):
        return self._layers(self._first_layer(z))


class Discriminator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self._layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=cfg['disc_dims'][i], out_features=cfg['disc_dims'][i + 1]),
                nn.LeakyReLU(negative_slope=.2, inplace=True),
            )
            for i in range(len(cfg['disc_dims']) - 1)
        ])
        self._last_layer = nn.Linear(in_features=cfg['disc_dims'][-1], out_features=1)

    def forward(self, x):
#        batch_std = x.view(x.shape[0] * x.shape[1], -1).std(0).mean().expand(x.shape[0], x.shape[1], 1)
#        x = torch.cat([x, batch_std], dim=-1)
        return self._last_layer(self._layers(x))
