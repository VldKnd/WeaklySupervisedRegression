import torch
import numpy as np

from src.utils import save_ckpt


def compute_metrics(data, test, data_scaler):
    pred, train, noise = data
    if data_scaler is not None:
        train = data_scaler.inverse_transform(train)
        pred = data_scaler.inverse_transform(pred)
    pred = abs(pred)
    metrics = {'kendall_test': absolute_kendall(test, pred), 'anderson_test': anderson_darling(test, pred),
               'kendall_train': absolute_kendall(train, pred), 'anderson_train': anderson_darling(train, pred)}
    return pred, noise, metrics


def update_metrics(metrics, meters):
    for m in metrics:
        for k, v in m.items():
            meters[k].update(v)


def update_best_metrics(meters, best_metrics, train_params, net_vars):
    for k, v in best_metrics.items():
        if v is None or meters[k].avg < v:
            best_metrics[k] = meters[k].avg
            save_ckpt(*train_params, best_metrics, k, net_vars)


def anderson_darling(sample, synthetic):
    """
    params: sample, synthetic - 2d numpy arrays of the same shape
    (row for day, column for stock index)
    """
    assert sample.shape == synthetic.shape
    n = synthetic.shape[0]
    ordered_syn = np.sort(synthetic, axis=0)
    U = np.array([(1 + np.sum(np.where(sample - row <= 0, 1, 0), axis=0)) / (n + 2) for row in ordered_syn])
    weights = np.array([2 * i - 1 for i in range(1, n + 1)])
    part1 = np.log(U)
    part2 = np.log(1 - np.flip(U, axis=0))
    return np.mean(-1 * n - weights @ (part1 + part2) / n)


def absolute_kendall(sample, synthetic):
    """
    params: sample, synthetic - 2d numpy arrays of the same shape
    (row for day, column for stock index)
    """
    assert sample.shape == synthetic.shape
    n = sample.shape[0]
    sorted_z = np.array(sorted(
        [np.sum(np.prod(np.where(sample - row < 0, 1, 0), axis=1)) / (n - 1) for row in sample]
    ))
    sorted_z_tild = np.array(sorted(
        [np.sum(np.prod(np.where(synthetic - row < 0, 1, 0), axis=1)) / (n - 1) for row in synthetic]
    ))
    return np.mean(np.abs(sorted_z - sorted_z_tild))


def anderson_darling_differentiable(real, fake, device):
    n = real.shape[0]
    fake_, _ = torch.sort(fake, 0)
    u = fake_.repeat_interleave(n, 0) - (real.repeat(n, 1) - 1e-6)
    u = torch.sigmoid(100 * u / (torch.abs(u) + 1e-6))
    u = (1 + torch.sum(u.view(n, -1, real.shape[1]), 0)) / (n + 2)
    w = torch.tensor([2. * i - 1 for i in range(1, n + 1)], dtype=torch.float, device=device)
    return -n - w / 2 @ (torch.log(u) + torch.log(1 - torch.flip(u, [0]))) / n


def absolute_kendall_differentiable(real, fake):
    def f(x):
        n = x.shape[1]
        x = x.repeat_interleave(n, 1) - (x.repeat((1, n, 1)) + 1e-6)
        x = torch.sigmoid(100 * x / (torch.abs(x) + 1e-6)).prod(1)
        x_, _ = torch.sort(x.view(x.shape[0], -1).sum(0) / (n - 1))
        return x_
    return torch.abs(f(real) - f(fake))
