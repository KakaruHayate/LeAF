import argparse
import random
import pathlib
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from dataset import LeAFTrainingDataset, LeAFValidationDataset, leaf_val_collate_fn
from optimizer.muon import Muon_AdamW

from mel_vit import MelViTModel
from module import Embedder, ARPredictor, SIGReg
from model import LeAF, Decoder

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
    im_saliency = ax1.imshow(saliency_norm, aspect='auto', origin='lower', cmap='hot', alpha=0.5)
    
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

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image

# ==========================================
# 2. 验证循环 (Validation Loop)
# ==========================================

def validate_step(dataloader, model, device, writer, global_step, use_decoder):
    """执行完整的验证并绘制热区图"""
    model.eval()

    sum_pred_mse = 0.0
    sum_mae, sum_mse = 0.0, 0.0
    gt_cache, pred_cache = [], []
    
    criterion_mse = nn.MSELoss()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(dataloader, desc='Validation', leave=False)):
            # Dataset 返回: (B, T, mels), 我们需转换成 LeAF 期待的 (B, 1, mels, T)
            mel_gt = batch['clean_mel'].transpose(1, 2).unsqueeze(1).to(device)
            pitch_gt = batch['pitch'].to(device)
            opec_gt = batch['opec'].to(device)
            
            info = {'mel': mel_gt, 'action': pitch_gt}
            
            # 模型前向
            emb, preds, curve_pred = model(info, mode='train')
            
            # Predictor 任务指标
            pred_mse = criterion_mse(preds[:, :-1, :], emb[:, 1:, :])
            sum_pred_mse += pred_mse.item() * mel_gt.size(0)

            # Decoder 任务指标
            if use_decoder:
                l_gt = model.decoder.normalize(opec_gt)
                loss_mse = criterion_mse(curve_pred.squeeze(-1), l_gt)
                sum_mse += loss_mse.item() * mel_gt.size(0)
                
                # 反归一化后计算 MAE 与 R2/Pearson 缓存
                y_pred = model.decoder.denormalize(curve_pred.squeeze(-1))
                sum_mae += F.l1_loss(y_pred, opec_gt).item() * mel_gt.size(0)
                
                gt_cache.append(opec_gt)
                pred_cache.append(y_pred)

    dataset_len = len(dataloader.dataset)
    mean_pred_mse = sum_pred_mse / dataset_len
    writer.add_scalar('val/Predictor_MSE', mean_pred_mse, global_step)
    print(f" --- [Validation] Predictor MSE: {mean_pred_mse:.6f}")

    if use_decoder:
        mean_mse = sum_mse / dataset_len
        mean_mae = sum_mae / dataset_len
        
        # 将变长序列拍平后计算相关性
        gt_all = torch.cat([g.view(-1) for g in gt_cache], dim=0)
        pred_all = torch.cat([p.view(-1) for p in pred_cache], dim=0)
        
        r2 = calc_r_squared(gt_all, pred_all)
        pearson = calc_pearson(gt_all, pred_all)
        
        writer.add_scalar('val/OPEC_MSE', mean_mse, global_step)
        writer.add_scalar('val/OPEC_MAE', mean_mae, global_step)
        writer.add_scalar('val/OPEC_R2', r2, global_step)
        writer.add_scalar('val/OPEC_Pearson', pearson, global_step)
        print(f" --- [Validation] OPEC MAE: {mean_mae:.4f} | R2: {r2:.4f} | Pearson: {pearson:.4f}")

    # ==========================================
    # 绘制可解释性热区图 (Saliency Map) - 仅取 Batch 第一个样本
    # ==========================================
    model.zero_grad()
    mel_sample = mel_gt[0:1].clone().detach().requires_grad_(True)
    pitch_sample = pitch_gt[0:1]
    opec_sample = opec_gt[0]

    with torch.enable_grad():
        info_vis = {'mel': mel_sample, 'action': pitch_sample}
        emb_v, preds_v, curve_pred_v = model(info_vis, mode='train')
        
        # 构建标量 Target 进行反传求导
        if use_decoder:
            target_value = curve_pred_v.sum()
        else:
            target_value = preds_v.sum()
            
        target_value.backward()
        saliency_map = mel_sample.grad[0, 0].cpu().numpy()

    mel_np = mel_sample[0, 0].detach().cpu().numpy()
    
    if use_decoder:
        gt_np = opec_sample.cpu().numpy()
        # 反归一化方便对比
        pred_np = model.decoder.denormalize(curve_pred_v[0].squeeze(-1)).detach().cpu().numpy()
        fig = plot_composite_saliency_map(mel_np, saliency_map, gt_np, pred_np)
    else:
        fig = plot_composite_saliency_map(mel_np, saliency_map)

    img = fig_to_image(fig)
    writer.add_image('val/saliency_overlay', np.array(img).transpose(2, 0, 1), global_step)
    model.train() # 恢复训练模式


# ==========================================
# 3. 主训练流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LeAF / World Model SVS Training")
    parser.add_argument('--exp_name', '-N', type=str, default="LeAF_Baseline")
    parser.add_argument('--dataset', '-d', required=True, help="Path to processed dataset")
    
    # 网络架构参数
    parser.add_argument('--use_decoder', action='store_true', help="If set, enable Decoder to predict OPEC")
    parser.add_argument('--hidden_size', type=int, default=192)
    parser.add_argument('--vit_layers', type=int, default=12)
    parser.add_argument('--vit_heads', type=int, default=3)
    parser.add_argument('--pred_depth', type=int, default=6)
    parser.add_argument('--pred_heads', type=int, default=64)
    parser.add_argument('--decoder_expansion', type=int, default=2)
    
    # 训练循环参数
    parser.add_argument('--batchsize', '-B', type=int, default=16)
    parser.add_argument('--cropsize', '-C', type=int, default=160)
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--save_val_interval', type=int, default=5000, help="Steps between save and validation")
    
    # 优化器与调度器参数
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_step_size', type=int, default=5000)
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
    writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")
    os.makedirs(f"./checkpoints/{args.exp_name}", exist_ok=True)

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
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False, 
        collate_fn=leaf_val_collate_fn, num_workers=2
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
        num_frames=3, depth=args.pred_depth, heads=args.pred_heads,
        mlp_dim=args.hidden_size*4, input_dim=args.hidden_size, hidden_dim=args.hidden_size
    )
    
    decoder = None
    if args.use_decoder:
        decoder = Decoder(hidden_size=args.hidden_size, out_dim=1, expansion=args.decoder_expansion)
        
    model = LeAF(encoder, predictor, action_encoder, decoder=decoder).to(device)
    
    # 构建损失函数
    sigreg_criterion = SIGReg(knots=17, num_proj=1024).to(device)
    huber_loss = nn.HuberLoss(reduction='none') # 使用 none 以便计算指数权重

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
    # 4. Step 驱动的主训练循环
    # -----------------------
    global_step = 0
    model.train()
    
    # 循环控制直到达到 max_steps
    train_iter = iter(train_loader)
    
    pbar = tqdm.tqdm(total=args.max_steps, desc="Training")
    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        # 数据转置与挂载: Dataset的 (B, T, mel) -> (B, 1, mel, T)
        mel_in = batch['aug_mel'].transpose(1, 2).unsqueeze(1).to(device)
        pitch_in = batch['pitch'].to(device)
        opec_gt = batch['opec'].to(device)
        
        info = {'mel': mel_in, 'action': pitch_in}
        
        optimizer.zero_grad()
        
        emb, preds, curve_pred = model(info, mode='train')
        
        # 1. 均方误差预测损失 (World Model 核心)
        loss_pred = F.mse_loss(preds[:, :-1, :], emb[:, 1:, :])
        
        # 2. SIGReg 表征防止坍缩损失
        loss_sigreg = sigreg_criterion(emb.transpose(0, 1))
        
        # 3. Decoder 任务损失 (若开启)
        loss_curve = torch.tensor(0.0, device=device)
        if args.use_decoder:
            l_gt = model.decoder.normalize(opec_gt)
            # 根据 GT 绝对值大小给予指数平滑权重，重点关注动作幅度大的区域
            sigma = 0.1
            weights = torch.exp(-l_gt.abs() / sigma).unsqueeze(-1)
            raw_huber = huber_loss(curve_pred, l_gt.unsqueeze(-1))
            loss_curve = (weights * raw_huber).mean()
            
        # 联合优化
        total_loss = loss_pred + args.lambd * loss_sigreg + loss_curve
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if global_step % 50 == 0:
            writer.add_scalar("Train/Total_Loss", total_loss.item(), global_step)
            writer.add_scalar("Train/Predictor_MSE", loss_pred.item(), global_step)
            writer.add_scalar("Train/SIGReg", loss_sigreg.item(), global_step)
            if args.use_decoder:
                writer.add_scalar("Train/OPEC_Huber", loss_curve.item(), global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            
        global_step += 1
        pbar.update(1)
        
        # ==========================================
        # 5. 定期 Save 与 Validation (Fixed Step)
        # ==========================================
        if global_step % args.save_val_interval == 0:
            print(f"\n--- Saving & Validation at Step {global_step} ---")
            
            # 先保存模型 (Save)
            checkpoint_path = f"./checkpoints/{args.exp_name}/model_step_{global_step}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # 后执行验证 (Validate)
            validate_step(val_loader, model, device, writer, global_step, args.use_decoder)

    pbar.close()
    writer.close()
    print("Training Complete!")

if __name__ == '__main__':
    main()