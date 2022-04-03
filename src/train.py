from shutil import rmtree
from functools import partial
from multiprocessing import Pool

import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import MyDataset
from src.gan import Generator, Discriminator
from src.metrics import (
    update_metrics,
    compute_metrics,
    update_best_metrics,
)
from src.utils import (
    set_seed,
    root_dir,
    save_ckpt,
    load_ckpt,
    log_meters,
    reset_meters,
    create_meters,
    print_cfg_info,
    load_data_scaler,
)


def train(epoch, nets, optims, datasets, logger, best_metrics, cfg):
    netG, netD = nets
    optimG, optimD = optims
    train_loader, test_dataset = datasets
    meter_names = ['kendall_test', 'anderson_test', 'kendall_train', 'anderson_train', 'eps_loss',
                   'netD_loss', 'netG_loss', 'grad_loss', 'netD_real_loss', 'netD_fake_loss']
    meters = create_meters(meter_names)
    best_metrics = {n: None for n in meter_names[:4]} if best_metrics is None else best_metrics
    data_scaler = load_data_scaler(cfg)
    func_compute = partial(compute_metrics, test=test_dataset[0], data_scaler=data_scaler)
    for e in range(epoch, epoch + len(train_loader)):
        reset_meters(meters)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            real = batch.to(cfg['device'], non_blocking=True)

            # discriminator
            optimD.zero_grad()
            with torch.no_grad():
                noise = torch.randn([len(real), cfg['mini_batch_size'], cfg['z_dim']], device=cfg['device'])
                fake = netG(noise)
            e_fake = netD(fake).mean()
            e_real = netD(real).mean()
            netD_loss = e_fake - e_real
            meters['netD_real_loss'].update(e_real.item())
            meters['netD_fake_loss'].update(e_fake.item())
            # eps
            eps_pen = e_real ** 2
            eps_loss = eps_pen.mean()
            meters['eps_loss'].update(eps_loss.item())
            # grad penalty
            mix_alpha = torch.rand((len(real), cfg['mini_batch_size'], 1), device=cfg['device'])
            mixed = mix_alpha * real + (1 - mix_alpha) * fake
            mixed.requires_grad_(True)
            mixed_preds = netD(mixed).sum()
            grads = torch.autograd.grad(
                outputs=mixed_preds,
                inputs=mixed,
                create_graph=True,
                retain_graph=True,
            )[0]
            grads = grads.view(len(real), cfg['mini_batch_size'], -1)
            grad_loss = ((grads.norm(2, dim=-1) - 1) ** 2).mean()
            meters['grad_loss'].update(grad_loss.item())
            # total discriminator loss
            netD_total_loss = netD_loss + cfg['w_gp'] * grad_loss + cfg['w_eps'] * eps_loss
            meters['netD_loss'].update(netD_total_loss.item())
            netD_total_loss.backward()
            optimD.step()

            # generator
            optimG.zero_grad()
            optimD.zero_grad()
            fake = netG(noise)
            netG_pred = netD(fake)
            netG_loss = -netG_pred.mean()
            meters['netG_loss'].update(-netG_loss.item())
            netG_loss.backward()
            optimG.step()

            if i % cfg['metrics_freq'] == 0:
                for k in best_metrics.keys():
                    meters[k].reset()
                real = real.cpu().numpy()
                fake = fake.detach().cpu().numpy()
                noise = noise.detach().cpu().numpy()
                with Pool(processes=cfg['num_wrks']) as pool:
                    results = pool.map(func_compute, zip(fake, real, noise))
                pred, noise, metrics = zip(*results)
                update_metrics(metrics, meters)
                update_best_metrics(meters, best_metrics, [e, nets, optims, cfg], [pred[0], noise[0]])
            if i % cfg['display_freq'] == 0:
                t = e * len(train_loader) + i
                log_meters(logger, meters, t)
                logger.flush()
                reset_meters(meters)
            pbar.desc = f'{e} - ' \
                        f'kendall test: {best_metrics["kendall_test"]:.5f} ' \
                        f'anderson test: {best_metrics["anderson_test"]:.5f} ' \
                        f'kendall train: {best_metrics["kendall_train"]:.5f} ' \
                        f'anderson train: {best_metrics["anderson_train"]:.5f}'

        save_ckpt(e, nets, optims, cfg, best_metrics)


def main(cfg):
    print_cfg_info(cfg)
    set_seed(cfg['seed'])

    train_dataset = MyDataset(cfg, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], drop_last=True,
                              pin_memory=cfg['pin_mem'], num_workers=cfg['num_wrks'])
    test_dataset = MyDataset(cfg, mode='test')
    netG = Generator(cfg).to(cfg['device'], non_blocking=True)
    netD = Discriminator(cfg).to(cfg['device'], non_blocking=True)
    optimG = Adam(filter(lambda p: p.requires_grad, netG.parameters()), betas=cfg['betas'], lr=cfg['lr'])
    optimD = Adam(filter(lambda p: p.requires_grad, netD.parameters()), betas=cfg['betas'], lr=cfg['lr'])

    epoch = 0
    best_metrics = None
    nets = [netG, netD]
    optims = [optimG, optimD]
    datasets = [train_loader, test_dataset]
    if cfg['from_ckpt']:
        epoch, nets, optims, best_metrics = load_ckpt(cfg['from_epoch'], nets, optims, cfg)
        epoch += 1
    else:
        rmtree(root_dir() / f'ckpt/{cfg["lbl"]}', ignore_errors=True)
    logger = SummaryWriter(root_dir() / f'ckpt/{cfg["lbl"]}')
    train(epoch, nets, optims, datasets, logger, best_metrics, cfg)


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        cfg_name = 'default'
    else:
        cfg_name = sys.argv[1]
    with open(root_dir() / f'cfg/{cfg_name}.json') as f:
        cfg = json.load(f)
    main(cfg)
