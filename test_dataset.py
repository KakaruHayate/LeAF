import os
import json
import shutil
import pathlib
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

# ==========================================
# 假设你的 dataset 代码保存在 leaf_datasets.py 中
# 请根据实际情况修改导入路径
# ==========================================
from dataset import LeAFTrainingDataset, LeAFValidationDataset, leaf_val_collate_fn

class TestLeAFDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        测试前的准备工作：构造 Mock 数据集
        创建临时目录并生成假的 npz、文本列表和 npy 文件
        """
        cls.mock_dir = pathlib.Path("./mock_leaf_data")
        cls.mock_dir.mkdir(parents=True, exist_ok=True)
        
        cls.n_mels = 80
        cls.crop_size = 128
        
        # 1. 生成 metadata.json
        with open(cls.mock_dir / 'metadata.json', 'w') as f:
            json.dump({"sample_rate": 16000, "mel_bins": cls.n_mels}, f)
            
        # 2. 生成 dummy npz 文件和 lengths
        # 模拟 5 个样本：3个给train，2个给valid。长度在 150 到 300 帧之间随机
        cls.train_files = ['train_0.npz', 'train_1.npz', 'train_2.npz']
        cls.valid_files = ['valid_0.npz', 'valid_1.npz']
        all_files = cls.train_files + cls.valid_files
        
        lengths = []
        for file_name in all_files:
            seq_len = np.random.randint(150, 300)
            lengths.append(seq_len)
            
            # 生成随机特征
            spectrogram = np.random.randn(seq_len, cls.n_mels).astype(np.float32)
            pitch = np.random.randn(seq_len).astype(np.float32)
            opec = np.random.randn(seq_len).astype(np.float32)
            
            np.savez(
                cls.mock_dir / file_name, 
                spectrogram=spectrogram, 
                pitch=pitch, 
                opec=opec
            )
            
        # 3. 生成 lengths.npy
        np.save(cls.mock_dir / 'lengths.npy', np.array(lengths[:len(cls.train_files)]))
        
        # 4. 生成 train.txt 和 valid.txt
        with open(cls.mock_dir / 'train.txt', 'w') as f:
            for name in cls.train_files:
                f.write(f"{name}\n")
                
        with open(cls.mock_dir / 'valid.txt', 'w') as f:
            for name in cls.valid_files:
                f.write(f"{name}\n")
                
        print("\n[Setup] 成功构造 Mock 数据集。")

    @classmethod
    def tearDownClass(cls):
        """
        测试完毕后的清理工作：删除临时 Mock 目录
        """
        if cls.mock_dir.exists():
            shutil.rmtree(cls.mock_dir)
            print("[Teardown] 已清理 Mock 数据集。")

    def test_training_dataset(self):
        """
        测试训练集 Dataset：
        1. 初始化是否成功
        2. 返回的 key 是否完整
        3. 裁剪后的 Tensor 形状是否严格等于 (crop_size, n_mels)
        """
        # 我们故意把增强概率设为 1.0，以确保增强管线里的代码都会被执行到，没有语法报错
        train_dataset = LeAFTrainingDataset(
            root_dir=self.mock_dir, 
            crop_size=self.crop_size,
            volume_aug_rate=1.0,
            noise_aug_rate=1.0,
            mask_aug_rate=1.0
        )
        
        # 测试 len
        self.assertGreater(len(train_dataset), 0, "训练集样本数量不应为 0")
        
        # 取出一条数据
        sample = train_dataset[0]
        
        # 检查 Keys
        expected_keys = {"clean_mel", "aug_mel", "pitch", "opec"}
        self.assertEqual(set(sample.keys()), expected_keys, "Training Dataset 返回的字典键值不匹配")
        
        # 检查类型和形状
        for key in expected_keys:
            self.assertIsInstance(sample[key], torch.Tensor, f"{key} 必须是 torch.Tensor")
            
        self.assertEqual(sample["clean_mel"].shape, (self.crop_size, self.n_mels))
        self.assertEqual(sample["aug_mel"].shape, (self.crop_size, self.n_mels))
        self.assertEqual(sample["pitch"].shape, (self.crop_size,))
        self.assertEqual(sample["opec"].shape, (self.crop_size,))

    def test_validation_dataset(self):
        """
        测试验证集 Dataset：
        1. 返回的特征是否是完整长度 (无裁剪)
        2. 确认没有无关的 key (如 aug_mel)
        """
        val_dataset = LeAFValidationDataset(root_dir=self.mock_dir)
        self.assertEqual(len(val_dataset), len(self.valid_files), "验证集文件数量读取错误")
        
        sample = val_dataset[0]
        expected_keys = {"clean_mel", "pitch", "opec"}
        self.assertEqual(set(sample.keys()), expected_keys)
        
        # 验证集不裁剪，形状的第一维(T)应该大于 0
        T, n_mels = sample["clean_mel"].shape
        self.assertGreater(T, 0)
        self.assertEqual(n_mels, self.n_mels)
        self.assertEqual(sample["pitch"].shape, (T,))

    def test_validation_collate_fn(self):
        """
        测试 DataLoader 与 collate_fn：
        1. 是否能正确补齐变长序列 (Padding)
        2. 返回的 batch tensor 形状是否为 (Batch, Max_T, Dim)
        3. lengths 张量是否正确记录了真实长度
        """
        val_dataset = LeAFValidationDataset(root_dir=self.mock_dir)
        
        # 将 batch_size 设为 2，正好能测出长度不一的两条数据如何被 pad
        val_loader = DataLoader(
            val_dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=leaf_val_collate_fn
        )
        
        batch = next(iter(val_loader))
        
        # 检查 batch keys
        expected_keys = {"clean_mel", "pitch", "opec", "lengths"}
        self.assertEqual(set(batch.keys()), expected_keys)
        
        # 检查 Batch 维度 (Batch_size, Max_T, Dim)
        max_t = batch["clean_mel"].shape[1]
        self.assertEqual(batch["clean_mel"].shape, (2, max_t, self.n_mels))
        self.assertEqual(batch["pitch"].shape, (2, max_t))
        
        # 检查 lengths 记录是否合法
        lengths = batch["lengths"]
        self.assertEqual(lengths.shape, (2,))
        self.assertTrue(torch.all(lengths <= max_t), "真实长度不应超过 Padding 后的 max_t")

if __name__ == '__main__':
    unittest.main()