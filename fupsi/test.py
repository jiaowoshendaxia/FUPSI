# -- coding:utf-8 --
# Evaluate the jointly trained prediction and super-resolution pipeline.

import os
import numpy as np
import argparse
from utils.metrics import get_MAE, get_MSE, get_MAPE
import torch

from prediction import TransAm
from UrbanSG import Generator
from data_process5 import get_dataloader_pre
from data_process import get_dataloader
import math


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

device = "cuda" if torch.cuda.is_available() else "cpu"
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
# Fixed coarse/fine scaling constants for MainSeed-RawCount-v2.
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--dataset', type=str, default='MainSeed_TaxiBJ_P4',
                    help='which dataset to use')
parser.add_argument('--ext_flag', type=parse_bool, default=False, help='whether to use external factors')
#prediction
parser.add_argument('--nb_flow', type=int, default=2)
parser.add_argument('--map_height', type=int, default=8)
parser.add_argument('--map_width', type=int, default=8)
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
                    help='dim of skip conv')
parser.add_argument('--day_len',type=int,default=48,choices=(24,48))
parser.add_argument('--seed', type=int, default=2024, help='random seed')

parser.add_argument('--test_pre_flag', type=parse_bool, default=False, help='whether to pretrain pre')
parser.add_argument('--lamda_s', type=float, default=0.1, help='weight of loss between high from prediction_out and high truth')
parser.add_argument('--lamda_p', type=float, default=0.01, help='weight of loss between prediction_out and prediction_truth')
parser.add_argument('--lr_pre', type=float, default=1e-6, help='adam: learning rate of prediction')
parser.add_argument('--lr_sr', type=float, default=1e-4, help='adam: learning rate of super resolution')


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

# test CUDA
cuda = True if torch.cuda.is_available() else False
"""
Evaluation routine.
"""
def test_pre(mode):
    model_path = 'saved_model/separate/{}/seed{}/cpt_noext/{}-{}-{}_{}_{}_{}'.format(opt.dataset, opt.seed,
                                                                     opt.len_closeness,
                                                                     opt.len_period,
                                                                     opt.len_trend,
                                                                     opt.n_heads,
                                                                     opt.num_layers,
                                                                    opt.skip_dim)
    model = TransAm(feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                    skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                    len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                    external_dim=opt.external_dim, dropout=opt.dropout,ext_flag=opt.ext_flag).to(device)
    model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))
    model.eval()
    save_path = os.path.join('data/{}'.format(opt.dataset), mode)
    os.makedirs(save_path, exist_ok=True)

    # Initialize evaluation accumulators.

    MSE = 100000
    datapath = os.path.join('data', opt.dataset)
    dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, 1, True, mode, map_H=opt.map_height, map_W=opt.map_width, day_len=opt.day_len, channel=opt.nb_flow)
    total_mse, total_mae, total_mape, total_rmse = 0, 0, 0, 0

    for n, (xc, xp, xt, ext,lable) in enumerate(dataloader):
        los = 0
        l = lable.to(device)
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        ext = ext.to(device)
        B, Tc, _, H, W = xc.shape
        with torch.no_grad():
            out = model(xc, xp, xt,ext)
            # Compute prediction error against the target.
        pre = out.cpu().detach().numpy() * opt.scaler_X
        l = l.cpu().detach().reshape(B,-1, H, W).numpy() * opt.scaler_X
        total_mse += get_MSE(pre, l)
        total_mae += get_MAE(pre, l)
        total_mape += get_MAPE(pre, l)

        if n == 0:
            test_result = pre.reshape(B,-1,H,W)
        else:
            test_result = np.concatenate((test_result,pre.reshape(B,-1,H,W)),axis=0)

    mse = total_mse / len(dataloader.dataset)
    mae = total_mae / len(dataloader.dataset)
    mape = total_mape / len(dataloader.dataset)
    rmse = np.sqrt(mse)

    f = open('{}/results.txt'.format(model_path), 'a')
    f.write("{}:\tRMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\n".format(mode,rmse, mae, mape))
    f.close()

    print('{} MSE = {:.6f} ,MAE = {:.6f}, MAPE = {:.6f},RMSE = {:.6f}'.format(mode,mse, mae, mape, rmse))
    np.save('{}/X_pref{}{}{}_{}_{}_{}.npy'.format(save_path,opt.len_closeness,opt.len_period,
                                               opt.len_trend,opt.n_heads,opt.num_layers,opt.skip_dim), np.array(test_result))
def test(mode = 'test'):
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
    netP = TransAm(in_channel=opt.nb_flow,feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           map_heigh=opt.map_height,map_width=opt.map_width,ext_flag=opt.ext_flag,external_dim=opt.external_dim, dropout=opt.dropout)
    netS = Generator(scale_factor=opt.upscale_factor, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
                     scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y, ext_flag=False, residual_flag=True, in_channel=opt.nb_flow)
    # load test set
    datapath = os.path.join('data', opt.dataset)
    dataloader = get_dataloader(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, opt.scaler_Y,1,opt.ext_flag,
        mode,map_H=opt.map_height,map_W=opt.map_width,day_len=opt.day_len,channel=opt.nb_flow)
    # save_path = 'test_result/to_stage/no_ext(f+r)'
    save_path = 'test_result/to_stage/no_ext(r)' # result save path
    os.makedirs(save_path, exist_ok=True)

    netP.load_state_dict(
        torch.load('{}/final_model.pt'.format(p_save_path)))
    netS.load_state_dict(torch.load('{}/final_model.pt'.format(g_save_path)))
    netP.eval()
    netS.eval()
    if cuda:
        netP.cuda()
        netS.cuda()
    np.set_printoptions(suppress=True)

    total_mse, total_mae, total_mape = 0, 0, 0
    pre_mse, pre_mae, pre_mape = 0, 0, 0

    for j, (xc, xp, xt, ext, next_lable, test_labels) in enumerate(dataloader):
    # for j, (xc, xp, xt, next_lable, test_labels) in enumerate(dataloader):
        B,l,_,H,W = xc.shape
        # ext = torch.zeros(1)
        next_lable=next_lable.reshape(B,-1,H,W)
        test_labels = test_labels.reshape(B,-1,H*opt.upscale_factor,W*opt.upscale_factor)
        if cuda:
            xc = xc.cuda()
            xp = xp.cuda()
            xt = xt.cuda()
            next_lable = next_lable.cuda()
            test_labels = test_labels.cuda()
            ext = ext.cuda()

        pre = netP(xc, xp, xt, ext)
        # pre = netP(xc, xp, xt)
        sr = netS(pre, ext)
        pre = pre.cpu().detach().numpy() * opt.scaler_X
        next_lable = next_lable.cpu().detach().numpy() * opt.scaler_X
        sr = sr.cpu().detach().numpy() * opt.scaler_Y
        test_labels = test_labels.cpu().detach().numpy() * opt.scaler_Y
        if j == 0:
            test_coarse = pre.reshape(B,-1,H,W)
            test_fine = sr.reshape(B,-1,H*opt.upscale_factor,W*opt.upscale_factor)
            true_coarse = next_lable.reshape(B,-1,H,W)
            true_fine = test_labels.reshape(B,-1,H*opt.upscale_factor,W*opt.upscale_factor)
        else:
            test_coarse = np.concatenate((test_coarse,pre.reshape(B,-1,H,W)),axis=0)
            true_coarse = np.concatenate((true_coarse,next_lable.reshape(B,-1,H,W)),axis=0)
            test_fine = np.concatenate((test_fine,sr.reshape(B,-1,H*opt.upscale_factor,W*opt.upscale_factor)),axis=0)
            true_fine = np.concatenate((true_fine,test_labels.reshape(B,-1,H*opt.upscale_factor,W*opt.upscale_factor)),axis=0)

        total_mse += get_MSE(sr, test_labels) * len(xc)
        total_mae += get_MAE(sr, test_labels) * len(xc)
        total_mape += get_MAPE(sr, test_labels) * len(xc)
        pre_mse += get_MSE(pre, next_lable) * len(xc)
        pre_mae += get_MAE(pre, next_lable) * len(xc)
        pre_mape += get_MAPE(pre, next_lable) * len(xc)

    rmse = np.sqrt(total_mse / len(dataloader.dataset))
    mae = total_mae / len(dataloader.dataset)
    mape = total_mape / len(dataloader.dataset)
    pre_rmse = np.sqrt(pre_mse / len(dataloader.dataset))
    pre_mae = pre_mae / len(dataloader.dataset)
    pre_mape = pre_mape / len(dataloader.dataset)

    np.save('{}/test_coarse.npy'.format(g_save_path), test_coarse)
    np.save('{}/true_coarse.npy'.format(g_save_path), true_coarse)
    np.save('{}/test_fine.npy'.format(g_save_path), test_fine)
    np.save('{}/true_fine.npy'.format(g_save_path), true_fine)

    f = open('{}/results.txt'.format(save_path), 'a')
    f.write("{}:RMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\tpre_rmse={:.6f}\t"
            "pre_mae={:.6f}\tpre_mape={:.6f}\n".format(mode, rmse, mae, mape, pre_rmse, pre_mae, pre_mape))
    f.close()
    print("{}:\tRMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\tpre_rmse={:.6f}\t"
          "pre_mae={:.6f}\tpre_mape={:.6f}\n".format(mode, rmse, mae, mape, pre_rmse, pre_mae, pre_mape))
    metrics_path = os.path.join(g_save_path, '{}_metrics.csv'.format(mode))
    with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
        metrics_file.write('dataset,seed,mode,RMSE,MAE,MAPE,RMSE_c,MAE_c,MAPE_c\n')
        metrics_file.write('{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
            opt.dataset, opt.seed, mode, rmse, mae, mape, pre_rmse, pre_mae, pre_mape))


    return


if __name__ == '__main__':
    if opt.test_pre_flag == True:
        test_pre('train')
        test_pre('valid')
        test_pre('test')
    else:
        # save_path = 'test_result/to_stage/no_ext(f+r)'
        save_path = 'test_result/to_stage/no_ext(r)'
        os.makedirs(save_path, exist_ok=True)
        f = open('{}/results.txt'.format(save_path), 'a')
        f.write('=' * 50)
        f.write('\n')
        f.write(
            'dataset:{}\tlamda_p:{}\tlamda_s:{}\tn_residuals:{}\tbase_channels:{}\tnum_epochs:{}\t'
            'len_closeness:{}\tlen_period:{}\tlen_trend:{}\tn_heads:{}\tnum_layers:{}\n'.format(opt.dataset,opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers))
        f.close()
        test('train')
        test('valid')
        test('test')
