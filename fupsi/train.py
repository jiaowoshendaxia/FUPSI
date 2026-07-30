# -- coding:utf-8 --
# Pretrain the predictor and then optimize prediction and super-resolution.
# Formal MainSeed experiments disable external factors.
import argparse
import csv
import os
import numpy as np
from math import log10,sqrt
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data
from torch import nn
from datetime import datetime
from utils.metrics import get_MAE, get_MSE, get_MAPE
import pytorch_ssim


def safe_ssim(img1, img2):
    try:
        return float(pytorch_ssim.ssim(img1, img2).item())
    except (TypeError, RuntimeError):
        mse = torch.mean((img1 - img2) ** 2)
        denom = torch.mean(img2 ** 2) + 1e-8
        return float(torch.clamp(1.0 - mse / denom, min=0.0, max=1.0).item())

from data_process5 import get_dataloader_pre
from data_process import get_dataloader
from prediction import TransAm
# from prediction1_3 import TransAm
from UrbanSG import Generator,Discriminator


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser = argparse.ArgumentParser(description='Train residual-enabled FUPSI')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor',choices=(2,4))
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--dataset', type=str, default='MainSeed_TaxiBJ_P4', help='processed dataset alias')
# Fixed coarse/fine scaling constants for MainSeed-RawCount-v2.
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--ext_flag', type=parse_bool, default=False, help='whether to use external factors')
#prediction
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=5)
parser.add_argument('--len_trend', type=int, default=0)
parser.add_argument('--external_dim', type=int, default=7)
parser.add_argument('--n_heads', type=int, default=4,
                    help='number of heads of selfattention')
parser.add_argument('--dim_head', type=int, default=8,
                    help='dim of heads of selfattention')
parser.add_argument('--dropout', type=float, default=0,
                    help='encoder dropout')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of encoder layers')
parser.add_argument('--feature_size', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dim of FC layer')
parser.add_argument('--skip_dim', type=int, default=128,
                    help='dim of skip conv',choices=(128,256))
parser.add_argument('--nb_flow', type=int, default=2,choices=(1,2))
parser.add_argument('--map_height', type=int, default=8)
parser.add_argument('--map_width', type=int, default=8)
parser.add_argument('--day_len',type=int,default=48,choices=(24,48))
# training skills
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate of prediction in pretrain')
parser.add_argument('--lamda_s', type=float, default=0.1, help='weight of loss between high from prediction_out and high truth')
parser.add_argument('--lamda_p', type=float, default=0.01, help='weight of loss between prediction_out and prediction_truth')
parser.add_argument('--lambda_adv', type=float, default=0.0, help='weight of the optional BCE adversarial loss')
parser.add_argument('--lr_pre', type=float, default=1e-6, help='adam: learning rate of prediction')
parser.add_argument('--lr_sr', type=float, default=1e-4, help='adam: learning rate of super resolution')
parser.add_argument('--lr_d', type=float, default=5e-6, help='adam: learning rate of discriminator')
parser.add_argument('--d_update_interval', type=int, default=20, help='update discriminator every N generator steps')
parser.add_argument('--real_label_smoothing', type=float, default=0.1, help='one-sided real-label smoothing')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--harved_epoch', type=int, default=20, help='halved at every x interval')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--train_pre_flag', type=parse_bool, default=False, help='whether to pretrain pre')
opt = parser.parse_args()


FORMAL_DATASET_CONFIGS = {
    "TaxiBJ_P1": (4, 8, 8, 48, 2),
    "TaxiBJ_P2": (4, 8, 8, 48, 2),
    "TaxiBJ_P3": (4, 8, 8, 48, 2),
    "TaxiBJ_P4": (4, 8, 8, 48, 2),
    "BikeNYC": (2, 8, 4, 24, 2),
}


def validate_formal_configuration():
    matched = next(
        (
            values
            for dataset_suffix, values in FORMAL_DATASET_CONFIGS.items()
            if opt.dataset.endswith(dataset_suffix)
        ),
        None,
    )
    if matched is None:
        return
    expected = dict(
        zip(
            ("upscale_factor", "map_height", "map_width", "day_len", "nb_flow"),
            matched,
        )
    )
    mismatches = [
        f"{name}={getattr(opt, name)} (expected {value})"
        for name, value in expected.items()
        if getattr(opt, name) != value
    ]
    if mismatches:
        parser.error(
            f"{opt.dataset} is inconsistent with MainSeed-RawCount-v2: "
            + "; ".join(mismatches)
        )


validate_formal_configuration()
print(opt)

def get_RMSE(pred, real):
    mse = np.mean(np.power(real - pred, 2))
    return sqrt(mse)

def train_pre(lr,epoch_num):
    # Fix all model-initialization random seeds.
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    rmses = [np.inf]
    save_path = 'saved_model/separate/{}/seed{}/cpt_noext/{}-{}-{}_{}_{}_{}'.format(opt.dataset, opt.seed,
                                                               opt.len_closeness,
                                                               opt.len_period,
                                                               opt.len_trend,
                                                               opt.n_heads,
                                                               opt.num_layers,
                                                                opt.skip_dim)
    os.makedirs(save_path, exist_ok=True)
    datapath = os.path.join('data', opt.dataset)

    train_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  opt.batch_size, True,
        mode='train', map_H=opt.map_height, map_W=opt.map_width, day_len=opt.day_len, channel=opt.nb_flow)  # opt.batch_size=16
    valid_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  4, True, mode='valid', map_H=opt.map_height, map_W=opt.map_width, day_len=opt.day_len, channel=opt.nb_flow)

    pre = TransAm(in_channel=opt.nb_flow,feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           map_heigh=opt.map_height,map_width=opt.map_width,ext_flag=opt.ext_flag,external_dim=opt.external_dim, dropout=opt.dropout)

    print('# prediction parameters:', sum(param.numel() for param in pre.parameters()))


    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        print("CUDA is available; training on GPU.")
        pre.cuda()
        criterion.cuda()

    optimizer = optim.Adam(pre.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    iter = 0
    for epoch in range(epoch_num):
        pre.train()
        train_loss = 0
        ep_time = datetime.now()
        """Run one pretraining epoch."""
        for z, (xc,xp,xt,ext,next) in enumerate(train_dataloader):

            optimizer.zero_grad()
            loss = 0
            B,Tc,_,H,W = xc.shape
            if torch.cuda.is_available():
                xc = xc.cuda()
                xp = xp.cuda()
                xt = xt.cuda()
                ext = ext.cuda()
                next = next.cuda()
            pred = pre(xc,xp,xt,ext)
            loss = criterion(pred, next.reshape(B,-1,H,W))
            loss.requires_grad_(True)
            # Update model parameters.
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                    epoch_num,
                                                                    z,
                                                                    len(train_dataloader),
                                                                    np.sqrt(loss.item())
                                                                    ))

            # counting training mse
            train_loss += loss.item()

            iter += 1
            # validation phase
            if iter % 20 == 0:
                with torch.no_grad():
                    pre.eval()
                    valid_time = datetime.now()
                    total_mse = 0
                    for n, (xc, xp, xt, ext, next) in enumerate(valid_dataloader):
                        los = 0
                        if torch.cuda.is_available():
                            xc = xc.cuda()
                            xp = xp.cuda()
                            xt = xt.cuda()
                            ext = ext.cuda()
                        Bv, Tv, _, H, W = xc.shape
                        pred = pre(xc, xp, xt, ext).cpu()
                        # Evaluate validation MSE.
                        los = criterion(pred, next.reshape(Bv, -1, H, W))
                        total_mse += los * Bv
                    rmse = np.sqrt(total_mse / len(valid_dataloader.dataset)) * opt.scaler_X
                    if rmse < np.min(rmses):
                        print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now() - valid_time))
                        torch.save(pre.state_dict(),
                                   '{}/final_model.pt'.format(save_path))
                    rmses.append(rmse)

        # half the learning rate
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            lr /= 2
            optimizer = optim.Adam(pre.parameters(), lr=lr)

        print('=================time cost: {}==================='.format(
            datetime.now() - ep_time))
    final_model_path = '{}/final_model.pt'.format(save_path)
    if not os.path.exists(final_model_path):
        torch.save(pre.state_dict(), final_model_path)
    return

def train(lr_pre,lr_sr):
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    rmses = [np.inf]
    #NoSpa/to_stage/RNN_FUFI
    p_save_path = 'saved_model/to_stage/no_ext(r)/{}/seed{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/cpt'.format(opt.dataset, opt.seed,
    # p_save_path = 'saved_model/NoSpa/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/cpt'.format(opt.dataset,
    # p_save_path = 'saved_model/to_stage/no_ext(f+r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/cpt'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    g_save_path = 'saved_model/to_stage/no_ext(r)/{}/seed{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Generator'.format(opt.dataset, opt.seed,
    # g_save_path = 'saved_model/NoSpa/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Generator'.format(opt.dataset,
    # g_save_path = 'saved_model/to_stage/no_ext(f+r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Generator'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    d_save_path = 'saved_model/to_stage/no_ext(r)/{}/seed{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Discriminator'.format(opt.dataset, opt.seed,
    # d_save_path = 'saved_model/NoSpa/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Discriminator'.format(opt.dataset,
    # d_save_path = 'saved_model/to_stage/no_ext(f+r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Discriminator'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    p_pretrain_path = 'saved_model/separate/{}/seed{}/cpt_noext/{}-{}-{}_{}_{}_{}'.format(opt.dataset, opt.seed,
    # p_pretrain_path = 'saved_model/separate/{}/seed{}/cpt_noext/{}-{}-{}_{}_{}_{}'.format(opt.dataset, opt.seed,
                                                               opt.len_closeness,
                                                               opt.len_period,
                                                               opt.len_trend,
                                                               opt.n_heads,
                                                               opt.num_layers,
                                                                opt.skip_dim)
    os.makedirs(p_save_path, exist_ok=True)
    os.makedirs(g_save_path, exist_ok=True)
    os.makedirs(d_save_path, exist_ok=True)
    valid_rmse = torch.zeros((opt.num_epochs,))
    datapath = os.path.join('data', opt.dataset)
    train_dataloader = get_dataloader(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, opt.scaler_Y, opt.batch_size,opt.ext_flag,
        'train',map_H=opt.map_height,map_W=opt.map_width,day_len=opt.day_len,channel=opt.nb_flow)  # opt.batch_size=16
    valid_dataloader = get_dataloader(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, opt.scaler_Y, 4, opt.ext_flag,
        'valid',map_H=opt.map_height,map_W=opt.map_width,day_len=opt.day_len,channel=opt.nb_flow)

    netP = TransAm(in_channel=opt.nb_flow,feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           map_heigh=opt.map_height,map_width=opt.map_width,ext_flag=opt.ext_flag,external_dim=opt.external_dim, dropout=opt.dropout)
    print('# predictor parameters:', sum(param.numel() for param in netP.parameters()))
    netS = Generator(scale_factor=opt.upscale_factor, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
                     scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y, ext_flag=False,residual_flag=True,in_channel=opt.nb_flow)
    # netS = Generator(scale_factor=UPSCALE_FACTOR, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
    #                  scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y, ext_flag=opt.ext_flag)

    print('# generator parameters:', sum(param.numel() for param in netS.parameters()))
    netD = Discriminator(in_channel=opt.nb_flow,ext_flag=False)
    # netD = Discriminator(ext_flag=opt.ext_flag)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    netP.load_state_dict(torch.load('{}/final_model.pt'.format(p_pretrain_path),map_location=torch.device('cpu')))
    criterion = nn.MSELoss()
    criterion_adv = nn.BCELoss()
    if torch.cuda.is_available():
        print("CUDA is available; training on GPU.")
        netP.cuda()
        netS.cuda()
        netD.cuda()
        criterion.cuda()

    optimizerG = optim.Adam([
        {'params': netP.parameters(), 'lr': lr_pre, 'betas': (0.9, 0.999)},
        {'params': netS.parameters(), 'lr': lr_sr, 'betas': (0.9, 0.999)},
    ])
    use_adversarial = opt.lambda_adv > 0
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2)) if use_adversarial else None
    print('adversarial regularization:', 'enabled' if use_adversarial else 'disabled')
    results = {
        'd_loss': [], 'g_loss': [], 'reconstruction_loss': [], 'adv_loss': [],
        'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'p_loss': []
    }
    min_rmse = 1000
    global_step = 0
    for epoch in range(1, opt.num_epochs + 1):
        train_bar = tqdm(train_dataloader)
        running_results = {
            'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0,
            'reconstruction_loss': 0, 'adv_loss': 0,
            'd_score': 0, 'g_score': 0
        }

        netP.train()
        netS.train()
        if use_adversarial:
            netD.train()
        out_path = '{}/valid_results/epoch{}_{}'.format(g_save_path,epoch, opt.num_epochs + 1)
        # Track intermediate predictions and losses.
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for xc, xp, xt, ext,pre, target in train_bar:
        # for xc, xp, xt, pre, target in train_bar:
            # Each batch contains aligned coarse and fine flow maps.
            batch_size ,Tc,_,H,W =  xc.shape
            pre = pre.reshape(batch_size,-1,H,W)
            #print("batch size = {}".format(batch_size))
            running_results['batch_sizes'] += batch_size
            #print("running_result = {}".format(running_results))

            ###########################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target.reshape(batch_size,-1,H*opt.upscale_factor,W*opt.upscale_factor)
            # print("real_img shape = {}".format(real_img.shape))
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            # print("z shape = {}".format(z.shape))
            # ext = torch.zeros(1)
            if torch.cuda.is_available():
                xc = xc.cuda()
                xp = xp.cuda()
                xt = xt.cuda()
                pre = pre.cuda()
                ext = ext.cuda()

            out_p = netP(xc,xp,xt,ext)
            # out_p = netP(xc, xp, xt)

            fake_img = netS(out_p, ext)
            loss_pre = criterion(out_p, pre)

            update_discriminator = use_adversarial and global_step % opt.d_update_interval == 0
            if update_discriminator:
                optimizerD.zero_grad()
                real_pred = netD(real_img, ext)
                fake_pred_detached = netD(fake_img.detach(), ext)
                real_target = torch.full_like(real_pred, 1.0 - opt.real_label_smoothing)
                d_loss_real = criterion_adv(real_pred, real_target)
                d_loss_fake = criterion_adv(fake_pred_detached, torch.zeros_like(fake_pred_detached))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                optimizerD.step()
            elif use_adversarial:
                with torch.no_grad():
                    real_pred = netD(real_img, ext)
                    fake_pred_detached = netD(fake_img, ext)
                    real_target = torch.full_like(real_pred, 1.0 - opt.real_label_smoothing)
                    d_loss_real = criterion_adv(real_pred, real_target)
                    d_loss_fake = criterion_adv(fake_pred_detached, torch.zeros_like(fake_pred_detached))
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
            else:
                d_loss = torch.zeros((), device=fake_img.device)

            ###########################
            # (2) Update G with supervised reconstruction and BCE adversarial regularization.
            ###########################
            optimizerG.zero_grad()
            reconstruction_loss = (
                opt.lamda_s * criterion(fake_img, real_img)
                + opt.lamda_p * loss_pre * opt.scaler_X / opt.scaler_Y
            )
            if use_adversarial:
                for parameter in netD.parameters():
                    parameter.requires_grad_(False)
                fake_pred_for_g = netD(fake_img, ext)
                adv_loss = criterion_adv(fake_pred_for_g, torch.ones_like(fake_pred_for_g))
                g_loss = reconstruction_loss + opt.lambda_adv * adv_loss
            else:
                fake_pred_for_g = None
                adv_loss = torch.zeros((), device=fake_img.device)
                g_loss = reconstruction_loss
            g_loss.backward()
            optimizerG.step()
            if use_adversarial:
                for parameter in netD.parameters():
                    parameter.requires_grad_(True)

            if use_adversarial:
                real_out = real_pred.mean().detach()
                fake_out = fake_pred_for_g.mean().detach()
            else:
                real_out = torch.zeros((), device=fake_img.device)
                fake_out = torch.zeros((), device=fake_img.device)

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['reconstruction_loss'] += reconstruction_loss.item() * batch_size
            running_results['adv_loss'] += adv_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            global_step += 1

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f ' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netP.eval()
        netS.eval()


        with torch.no_grad():
            iter = 0
            val_bar = tqdm(valid_dataloader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'p_loss':0}

            for val_xc, val_xp,val_xt,val_ext,val_next, val_hr in val_bar:
            # for val_xc, val_xp, val_xt,  val_next, val_hr in val_bar:
                batch_size ,Tc,_,H,W =  val_xc.shape
                val_next = val_next.reshape(batch_size,-1,H,W)
                valing_results['batch_sizes'] += batch_size
                val_hr= val_hr.reshape(batch_size,-1,H*opt.upscale_factor,W*opt.upscale_factor)
                # val_ext = torch.zeros(1)

                if torch.cuda.is_available():
                    val_xc = val_xc.cuda()
                    val_xp = val_xp.cuda()
                    val_xt = val_xt.cuda()
                    val_next = val_next.cuda()
                    val_hr = val_hr.cuda()
                    val_ext = val_ext.cuda()

                val_out_p = netP(val_xc, val_xp, val_xt, val_ext)
                # val_out_p = netP(val_xc, val_xp, val_xt)
                sr_p = netS(val_out_p,val_ext)

                if iter == 0:
                    pres = val_out_p.reshape(batch_size,-1, H, W)
                else:
                    pres = torch.cat((pres, val_out_p.reshape(batch_size, -1,H, W)), dim=0)
                iter += 1
                batch_ploss = get_MSE(val_out_p.cpu().detach().numpy(), val_next.cpu().numpy())

                batch_mse = ((sr_p - val_hr) ** 2).mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = safe_ssim(sr_p, val_hr)
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (val_hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                valing_results['p_loss'] += batch_ploss * batch_size
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f Loss_p:%.6f' % (
                        valing_results['psnr'], valing_results['ssim'],
                        valing_results['p_loss'] / valing_results['batch_sizes']))

            rmse = sqrt(valing_results['mse'] / len(valid_dataloader.dataset))
            valid_rmse[epoch-1]=rmse
            if rmse < min_rmse:
                min_rmse = rmse
                torch.save(netP.state_dict(), '{}/final_model.pt'.format(p_save_path))
                torch.save(netS.state_dict(), '{}/final_model.pt'.format(g_save_path))
                torch.save(netD.state_dict(), '{}/final_model.pt'.format(d_save_path))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['reconstruction_loss'].append(running_results['reconstruction_loss'] / running_results['batch_sizes'])
        results['adv_loss'].append(running_results['adv_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['p_loss'].append(valing_results['p_loss'] / valing_results['batch_sizes'])
        np.save(os.path.join(g_save_path, 'valid_rmse.npy'), valid_rmse.numpy())
        for key, values in results.items():
            np.save(os.path.join(g_save_path, '{}.npy'.format(key)), np.asarray(values, dtype=np.float64))
        history_path = os.path.join(g_save_path, 'training_history.csv')
        with open(history_path, 'w', encoding='utf-8', newline='') as history_file:
            fieldnames = ['epoch'] + list(results.keys()) + ['valid_rmse']
            writer = csv.DictWriter(history_file, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(len(results['g_loss'])):
                row = {'epoch': index + 1, 'valid_rmse': float(valid_rmse[index])}
                row.update({key: float(values[index]) for key, values in results.items()})
                writer.writerow(row)

    return

if __name__ == '__main__':
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    flag = opt.train_pre_flag
    if flag:
        train_pre(opt.lr,opt.num_epochs)
    else:
        train(opt.lr_pre,opt.lr_sr)
