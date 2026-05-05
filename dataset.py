import json
import pathlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset

 from augment.utils import (
     add_diverse_background_noise,
     frequency_masking,
     time_masking,
     time_frequency_noise_block
 )


class LeAFTrainingDataset(Dataset):
    def __init__(
            self,
            root_dir: pathlib.Path,
            crop_size: int = 160, 
            # 三种独立增强机制的触发概率
            volume_aug_rate: float = 0.5,
            noise_aug_rate: float = 0.3,
            mask_aug_rate: float = 0.4,
            # 当命中 Mask 增强时，三种 Mask (时间、频率、噪声块) 的分配比例
            # 顺序: [time_mask, freq_mask, block_mask]，要求和为 1.0
            mask_ratios: list = [0.2, 0.2, 0.6] 
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)

        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = json.load(f)

        self.root_dir = root_dir
        self.crop_size = crop_size
        
        self.volume_aug_rate = volume_aug_rate
        self.noise_aug_rate = noise_aug_rate
        self.mask_aug_rate = mask_aug_rate
        self.mask_ratios = mask_ratios

        # 读取文件列表
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())

        # 读取长度（帧数）
        self.lengths = np.load(root_dir / 'lengths.npy')
        if len(self.files) != len(self.lengths):
            raise ValueError("Elements in train.txt and lengths.npy do not match!")

        # 只保留长度 >= crop_size 的有效文件
        self.valid_indices = [i for i, l in enumerate(self.lengths) if l >= self.crop_size]
        if not self.valid_indices:
            raise ValueError("All data is too short for cropping!")

        self.valid_lengths = self.lengths[self.valid_indices]   # 有效帧数数组
        self.total_valid_frames = int(sum(self.valid_lengths))  # 总有效帧数

        # 每个 epoch 的采样次数（总帧数 // crop_size）
        self.epoch_samples = self.total_valid_frames // self.crop_size

    def __len__(self):
        return self.epoch_samples

    def apply_masking_roulette(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        按照设定的比例 (mask_ratios) 随机选择一种 Mask 策略进行应用
        """
        # 使用 random.choices 按照权重进行单次随机抽样
        mask_type = random.choices(
            population=['time', 'freq', 'block'],
            weights=self.mask_ratios,
            k=1
        )[0]

        if mask_type == 'time':
            return time_masking(mel_tensor, time_mask_param=3 num_masks=2)
        elif mask_type == 'freq':
            return frequency_masking(mel_tensor, freq_mask_param=3, num_masks=2)
        else:
            return time_frequency_noise_block(
                mel_tensor, 
                num_blocks=5, 
                max_time_ratio=0.2, 
                max_freq_ratio=0.15, 
                noise_mode='replace'
            )

    def __getitem__(self, idx):
        # 1. 按帧数加权随机抽样一个有效文件
        weights = self.valid_lengths / self.total_valid_frames
        i = np.random.choice(len(self.valid_indices), p=weights)

        file_idx = self.valid_indices[i]
        data = np.load(self.files[file_idx])
        
        # 读取 npz 中的核心特征
        spectrogram = data['spectrogram']    # (T, mel_bins)
        pitch = data['pitch']                # (T,)
        opec = data['opec']                  # (T,)

        # 2. 随机裁剪起点并同步切片
        T = spectrogram.shape[0]
        start = random.randint(0, T - self.crop_size)

        clean_mel_np = spectrogram[start:start + self.crop_size, :]
        crop_pitch = pitch[start:start + self.crop_size]
        crop_opec = opec[start:start + self.crop_size]

        # 3. 数据增强管线 (Augmentation Pipeline)
        aug_mel_np = clean_mel_np.copy()

        # [机制 1] 音量增强 (Volume Augmentation) - Numpy 域处理
        if random.random() < self.volume_aug_rate:
            aug_mel_np += np.random.uniform(-3, 3)

        # 全局数值底噪裁剪，防止异常极小值 (保留原始逻辑)
        aug_mel_np = np.clip(aug_mel_np, a_min=-12, a_max=None)
        clean_mel_np = np.clip(clean_mel_np, a_min=-12, a_max=None)

        # 将 Numpy 转为 Tensor，并调整形状为 (n_mels, T) 以适配增强函数
        # 这里的 clone() 是为了彻底脱离 numpy array 的底层内存关联
        clean_mel_tensor = torch.from_numpy(clean_mel_np).float().T
        aug_mel_tensor = torch.from_numpy(aug_mel_np).float().T

        # [机制 2] 物理底噪添加 (Background Noise) - Torch 域处理
        if random.random() < self.noise_aug_rate:
            aug_mel_tensor = add_diverse_background_noise(
                aug_mel_tensor, noise_level=0.001, noise_type='random'
            )

        # [机制 3] 谱图掩蔽 (Masking) - Torch 域处理
        if random.random() < self.mask_aug_rate:
            aug_mel_tensor = self.apply_masking_roulette(aug_mel_tensor)

        # 转换回模型常规期待的形状 (T, n_mels)
        clean_mel_tensor = clean_mel_tensor.T
        aug_mel_tensor = aug_mel_tensor.T

        # 将 pitch 和 opec 转为 tensor
        crop_pitch_tensor = torch.from_numpy(crop_pitch).float()
        crop_opec_tensor = torch.from_numpy(crop_opec).float()

        # 返回干净特征、增强特征以及对齐的标签（供后续 Loss 计算或 World Model 推理）
        return {
            "clean_mel": clean_mel_tensor,
            "aug_mel": aug_mel_tensor,
            "pitch": crop_pitch_tensor,
            "opec": crop_opec_tensor
        }


class LeAFValidationDataset(Dataset):
    def __init__(self, root_dir: pathlib.Path):
        """
        LEAF 项目验证集 DataLoader
        特点：加载完整的序列特征，不包含任何随机切片或数据增强。
        """
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
            
        self.root_dir = root_dir

        # 读取验证集文件列表
        self.files = []
        valid_list_path = root_dir / 'valid.txt'
        
        if not valid_list_path.exists():
            raise FileNotFoundError(f"Validation list not found at: {valid_list_path}")
            
        with open(valid_list_path, 'r', encoding='utf8') as f:
            for line in f:
                # 拼接完整的 npz 文件路径
                file_path = root_dir / line.strip()
                self.files.append(file_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. 读取对应索引的 npz 文件
        data = np.load(self.files[idx])
        
        # 2. 提取我们在预处理时保存的三个核心特征
        spectrogram = data['spectrogram']  # 形状: (T, mel_bins)
        pitch = data['pitch']              # 形状: (T,)
        opec = data['opec']                # 形状: (T,)
        
        # 3. 基础的数值稳定性处理 (与训练集保持相同的数值底线)
        spectrogram = np.clip(spectrogram, a_min=-12, a_max=None)
        
        # 4. 转换为 PyTorch Tensors
        # 注意：这里我们保留原始的 (T, n_mels) 形状，不进行转置
        # 因为后续的 Padding 和网络模型通常习惯 Batch First 的序列输入: (Batch, Time, Dim)
        mel_tensor = torch.from_numpy(spectrogram).float()
        pitch_tensor = torch.from_numpy(pitch).float()
        opec_tensor = torch.from_numpy(opec).float()
        
        # 5. 使用与训练集一致的字典形式返回
        return {
            "clean_mel": mel_tensor,
            "pitch": pitch_tensor,
            "opec": opec_tensor
        }
