import argparse
import random
import pathlib
import os
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt
import yaml

from dataset import LeAFTrainingDataset, LeAFValidationDataset
from optimizer.muon import Muon_AdamW

from mel_vit import MelViTModel
from module import Embedder, ARPredictor, SIGReg, MLP
from model import LeAF, Decoder
from convnext import ConvNeXtDecoder

import logger.utils
from logger import utils
from logger.saver import Saver

# ==========================================
# 1. 评估指标与可视化辅助函数
# ==========================================

def calc_r_squared(y_true, y_pred):
    ss_res = torch.sum((y_pred - y_true) ** 2)
    mean_y_true = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean_y_true) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else torch.tensor(0.0, device=y_true.device)
    return r2.item()

def calc_pearson(y_true, y_pred):
    vx = y_true - torch.mean(y_true)
    vy = y_pred - torch.mean(y_pred)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return cost.item()

def plot_composite_saliency_map(mel, saliency, gt_curve=None, pred_curve=None):
    """绘制复合可视化图：底层 Mel + 透明 Saliency + 顶层曲线(可选)"""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 1. 绘制底层：原始 Mel 频谱
    ax1.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    ax1.set_ylabel('Mel Bins')
    ax1.set_xlabel('Time Frames')

    # 2. 绘制中层：Saliency 热区图叠加
    saliency_norm = np.abs(saliency)
    saliency_norm = saliency_norm / (np.max(saliency_norm) + 1e-8)
    ax1.imshow(saliency_norm, aspect='auto', origin='lower', cmap='hot', alpha=0.5)
    
    # 3. 绘制顶层：GT 与 Pred 曲线 (若存在 Decoder)
    if gt_curve is not None and pred_curve is not None:
        ax2 = ax1.twinx()
        ax2.plot(gt_curve, color='cyan', linewidth=2.5, label='GT')
        ax2.plot(pred_curve, color='lawngreen', linewidth=2.5, linestyle='--', label='Pred')
        ax2.set_ylabel('Acoustic Parameter Value (OPEC)')
        ax2.legend(loc='upper right')
        
    plt.title('Mel Spectrogram + Saliency Overlay' + (' with GT & Pred' if gt_curve is not None else ''))
    plt.tight_layout()
    return fig

# ==========================================
# 2. 验证循环 (Validation Loop)
# ==========================================

def validate_step(dataloader, model, device, saver, use_decoder, draw=True):
    model.eval()

    sum_pred_mse = 0.0
    sum_mae = 0.0
    sum_mse = 0.0
    gt_cache = []
    pred_cache = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for idx, (mel_gt, pitch_gt, opec_gt) in enumerate(
                tqdm.tqdm(dataloader, total=len(dataloader), desc='Validation', leave=False)):
            
            # 转置并补充维度: (B, T, mels) -> (B, 1, mels, T)
            mel_in = mel_gt.transpose(1, 2).unsqueeze(1).to(device)
            #pitch_in = pitch_gt.to(device)
            opec_gt = opec_gt.to(device)
            
            #info = {'mel': mel_in, 'action': pitch_in}
            info = {'mel': mel_in}
            emb, preds, curve_pred = model(info, mode='train')
            
            # 1. Predictor 任务指标
            pred_mse = criterion(preds[:, :-1, :], emb[:, 1:, :])
            sum_pred_mse += pred_mse.item() * mel_gt.size(0)

            # 2. Decoder 任务指标与绘图
            if use_decoder:
                l_gt = model.decoder.normalize(opec_gt)
                loss_mse = criterion(curve_pred.squeeze(-1), l_gt)
                sum_mse += loss_mse.item() * mel_gt.size(0)
                
                y_pred = model.decoder.denormalize(curve_pred.squeeze(-1))
                gt_cache.append(opec_gt)
                pred_cache.append(y_pred)
                sum_mae += F.l1_loss(y_pred, opec_gt).item() * mel_gt.size(0)
                
                # 绘制当前样本的预测曲线对比图
                if draw:
                    spec_draw = mel_gt[0].cpu().numpy()      # (T, mels)
                    curve_gt_draw = opec_gt[0].cpu().numpy() # (T,)
                    curve_pred_draw = y_pred[0].cpu().numpy()# (T,)
                    
                    # 避免过长的序列导致图片比例失调
                    if spec_draw.shape[0] > 1024:
                        spec_draw = spec_draw[:1024]
                        curve_gt_draw = curve_gt_draw[:1024]
                        curve_pred_draw = curve_pred_draw[:1024]
                        
                    saver.log_figure({
                        f'val_curve_{idx}': logger.utils.draw_plot(
                            spec=spec_draw,
                            curve_gt=curve_gt_draw,
                            curve_pred=curve_pred_draw
                        )
                    })

    dataset_len = len(dataloader.dataset)
    mean_pred_mse = sum_pred_mse / dataset_len
    
    saver.log_value({'val/Predictor_MSE': mean_pred_mse})
    saver.log_info(f" --- [Validation] Predictor MSE: {mean_pred_mse:.6f}")

    if use_decoder:
        mean_mse = sum_mse / dataset_len
        mean_mae = sum_mae / dataset_len
        
        gt_all = torch.cat([g.view(-1) for g in gt_cache], dim=0)
        pred_all = torch.cat([p.view(-1) for p in pred_cache], dim=0)
        
        r2 = calc_r_squared(gt_all, pred_all)
        pearson = calc_pearson(gt_all, pred_all)
        
        saver.log_value({
            'val/OPEC_MSE': mean_mse,
            'val/OPEC_MAE': mean_mae,
            'val/OPEC_R2': r2,
            'val/OPEC_Pearson': pearson
        })
        saver.log_info(f" --- [Validation] OPEC MAE: {mean_mae:.4f} | R2: {r2:.4f} | Pearson: {pearson:.4f}")

    model.train()


# ==========================================
# 3. 主训练流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LeAF / World Model SVS Training")
    parser.add_argument('--exp_name', '-N', type=str, default="LeAF_Baseline")
    parser.add_argument('--dataset', '-d', required=True, help="Path to processed dataset")
    parser.add_argument('--pretrained_model', '-P', type=str, default=None, help="Path to pretrained checkpoint to load")
    
    # 网络架构参数
    parser.add_argument('--use_decoder', default=True, action='store_true', help="If set, enable Decoder to predict OPEC")
    parser.add_argument('--hidden_size', type=int, default=192)
    parser.add_argument('--vit_layers', type=int, default=12)
    parser.add_argument('--vit_heads', type=int, default=3)
    parser.add_argument('--pred_depth', type=int, default=6)
    parser.add_argument('--pred_heads', type=int, default=64)
    parser.add_argument('--decoder_expansion', type=int, default=4)
    parser.add_argument('--skip_steps', type=int, default=40, help="Number of frames to skip for prediction target")
    
    # 训练循环参数
    parser.add_argument('--batchsize', '-B', type=int, default=16)
    parser.add_argument('--cropsize', '-C', type=int, default=160)
    parser.add_argument('--epoch', '-E', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--save_val_interval', type=int, default=4000, help="Steps between save and validation")
    
    # 优化器与调度器参数
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_step_size', type=int, default=4000)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--lambd', type=float, default=0.1, help="SIGReg scale factor")
    
    # 数据增强参数
    parser.add_argument('--vol_aug', type=float, default=0.5)
    parser.add_argument('--noise_aug', type=float, default=0.2)
    parser.add_argument('--mask_aug', type=float, default=0.2)
    
    args = parser.parse_args()
    
    # 固定随机种子
    seed = 3047
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 Logger
    saver = Saver(args.exp_name)

    # -----------------------
    # 1. 准备 DataLoader
    # -----------------------
    train_dataset = LeAFTrainingDataset(
        root_dir=args.dataset,
        crop_size=args.cropsize,
        volume_aug_rate=args.vol_aug,
        noise_aug_rate=args.noise_aug,
        mask_aug_rate=args.mask_aug
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    val_dataset = LeAFValidationDataset(root_dir=args.dataset)
    
    # 修改这里：batch_size 改为 1，去掉 collate_fn，num_workers 设为 0 以防死锁
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # -----------------------
    # 2. 构建模型
    # -----------------------
    encoder = MelViTModel(
        n_mels=80, in_channels=1, hidden_size=args.hidden_size, 
        num_hidden_layers=args.vit_layers, num_attention_heads=args.vit_heads
    )
    action_encoder = Embedder(input_dim=1, emb_dim=args.hidden_size)
    predictor = ARPredictor(
        num_frames=8096, depth=args.pred_depth, heads=args.pred_heads,
        mlp_dim=args.hidden_size*4, input_dim=args.hidden_size, hidden_dim=args.hidden_size
    )
    
    decoder = None
    if args.use_decoder:
        decoder = ConvNeXtDecoder(args.hidden_size, 1)
        
    projector = MLP(
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        hidden_dim=args.hidden_size*4,
        norm_fn=nn.LayerNorm,
    )

    predictor_proj = MLP(
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        hidden_dim=args.hidden_size*4,
        norm_fn=nn.LayerNorm,
    )
    
    model = LeAF(encoder, predictor, action_encoder, projector=projector, pred_proj=predictor_proj, decoder=decoder).to(device)
    
    params_count = utils.get_network_paras_amount({'model': model})
    print(model)
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    sigreg_criterion = SIGReg(knots=17, num_proj=1024).to(device)
    huber_loss = nn.HuberLoss(reduction='none')

    # -----------------------
    # 3. 优化器与调度器
    # -----------------------
    optimizer = Muon_AdamW(
        model, 
        lr=args.lr,
        muon_args={'weight_decay': args.weight_decay}, 
        adamw_args={'weight_decay': 0}
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # -----------------------
    # 4. 恢复状态 / 加载预训练模型
    # -----------------------
    start_epoch = 0
    global_step = 0
    
    # 优先检查是否存在当前 exp_name 的续训权重
    ckpt_files = glob.glob(str(saver.exp_dir / "checkpoint_*.pt"))
    if len(ckpt_files) > 0:
        latest_ckpt = max(ckpt_files, key=lambda x: int(re.search(r'checkpoint_(\d+)\.pt', x).group(1)))
        saver.log_info(f"Found existing experiment! Resuming training from: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        global_step = ckpt['global_step']
        saver.global_step = global_step
    # 否则检查是否传入了外部预训练权重
    elif args.pretrained_model is not None:
        saver.log_info(f"Loading external pretrained model: {args.pretrained_model}")
        ckpt = torch.load(args.pretrained_model, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        # strict=False 允许加载缺失/冗余 decoder 的情况
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        saver.log_info(f"Pretrained missing keys (e.g. new decoder layers): {missing}")
        saver.log_info(f"Pretrained unexpected keys: {unexpected}")

    # -----------------------
    # 5. Epoch 驱动的进度条主训练循环
    # -----------------------
    model.train()
    
    for epoch in range(start_epoch, args.epoch):
        if global_step >= args.max_steps:
            break
            
        # 设置 tqdm 进度条，包含 epoch 提示
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch}", leave=True)
        
        for batch in pbar:
            # 兼容 logger 的 global_step 并与自定义 global_step 同步
            saver.global_step_increment()
            global_step = saver.global_step
            
            mel_in = batch['aug_mel'].transpose(1, 2).unsqueeze(1).to(device)
            #pitch_in = batch['pitch'].to(device)
            opec_gt = batch['opec'].to(device)
            # 修正后的代码
            #log_pitch = (1 + pitch_in / 700).log()            # (B, T)
            # 如果 pitch_in 是 (B, T, 1)，可以先 squeeze(-1) 变成 (B, T)
            #if log_pitch.dim() == 3:
            #    log_pitch = log_pitch.squeeze(-1)
            #
            # 计算一阶差分（沿时间轴）
            #delta_pitch = log_pitch[:, 1:] - log_pitch[:, :-1]  # (B, T-1)
            #
            # 补齐第一帧（可选：用0填充，或用原始第一帧值）
            #first_frame = torch.zeros_like(delta_pitch[:, :1])  # 第一帧补0，形状 (B, 1)
            #delta_pitch = torch.cat([first_frame, delta_pitch], dim=1)  # (B, T)
            #
            # 恢复为三维以适应 Embedder 的输入要求 (B, T, 1)
            #delta_pitch = delta_pitch.unsqueeze(-1)              # (B, T, 1)

            # 然后传入 info
            # info = {'mel': mel_in, 'action': delta_pitch}
            info = {'mel': mel_in}
            
            optimizer.zero_grad()
            skip_steps = args.skip_steps  # 跳跃步数，可以设为超参数

            emb, preds, curve_pred = model(info, mode='train')
            loss_pred = F.mse_loss(preds[:, :-skip_steps, :], emb[:, skip_steps:, :])
            
            loss_sigreg = sigreg_criterion(emb.transpose(0, 1))
            
            loss_curve = torch.tensor(0.0, device=device)
            if args.use_decoder:
                l_gt = model.decoder.normalize(opec_gt)
                loss_curve = huber_loss(curve_pred, l_gt.unsqueeze(-1)).mean()
                
            total_loss = loss_pred + args.lambd * loss_sigreg + loss_curve
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            
            # 更新进度条右侧状态面板
            pbar_dict = {
                'loss': f"{total_loss.item():.4f}", 
                'pred_mse': f"{loss_pred.item():.4f}",
                'lr': f"{current_lr:.2e}"
            }
            if args.use_decoder:
                pbar_dict['opec_huber'] = f"{loss_curve.item():.4f}"
            pbar.set_postfix(pbar_dict)
            
            # Logging
            if global_step % 50 == 0:
                saver.log_value({
                    "Train/Total_Loss": total_loss.item(),
                    "Train/Predictor_MSE": loss_pred.item(),
                    "Train/SIGReg": loss_sigreg.item(),
                    "Train/LR": current_lr
                })
                if args.use_decoder:
                    saver.log_value({"Train/OPEC_Huber": loss_curve.item()})
            
            # ==========================================
            # 按照 Step 保存权重与验证
            # ==========================================
            if global_step % args.save_val_interval == 0:
                saver.log_info(f"\n--- Saving & Validation at Step {global_step} ---")
                
                # 统一打包当前所有状态以供无缝续训
                ckpt_path = saver.exp_dir / f"checkpoint_{global_step}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step
                }, ckpt_path)
                
                validate_step(val_loader, model, device, saver, args.use_decoder)
                
            if global_step >= args.max_steps:
                break

    saver.log_info("Training Complete!")

if __name__ == '__main__':
    main()