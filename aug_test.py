import os
import torch
import torchaudio
import matplotlib.pyplot as plt

# 从你的项目模块中导入特征提取与增强算法
# 请确保当前运行路径可以正确解析这两个模块
from r3moe.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch
from augment.utils import (
    add_physical_background_noise,
    frequency_masking,
    time_masking,
    time_frequency_noise_block,
    add_diverse_background_noise
)

def plot_and_save_comparison(original_mel, augmented_mel, title, save_path):
    """
    绘制原始 Mel 与增强后 Mel 的对比图并保存
    """
    # 转换为 numpy，并去掉 batch 维度 [n_mels, time]
    orig_np = original_mel.squeeze().cpu().numpy()
    aug_np = augmented_mel.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制原始谱图
    im1 = axes[0].imshow(orig_np, aspect='auto', origin='lower', interpolation='none', cmap='magma')
    axes[0].set_title("Original Log-Mel Spectrogram")
    axes[0].set_ylabel("Mel Bins")
    fig.colorbar(im1, ax=axes[0], format='%+2.0f dB')

    # 绘制增强后的谱图
    im2 = axes[1].imshow(aug_np, aspect='auto', origin='lower', interpolation='none', cmap='magma')
    axes[1].set_title(f"Augmented: {title}")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Bins")
    fig.colorbar(im2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {save_path}")

def test_data_augmentations(wav_path="test_sample.wav", output_dir="./aug_outputs"):
    """
    运行四种数据增强的单元测试
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running test on device: {device}")

    # 1. 初始化 Mel 提取器 (参考官方预处理脚本的参数设置)
    mel_extractor = PitchAdjustableMelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        f_min=0,
        f_max=None,
        n_mels=80,
        center=True  # 注意这里：参考官方处理流程，将 center 改为 True
    )

    # 2. 读取音频并统一采样率
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Please provide a valid wav file path. Could not find: {wav_path}")
        
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    
    # 确保音频为单声道，且形状严格为 [1, time] (Batch, Time)
    if wav.dim() == 2:
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
        
    wav = wav.to(device)

    # 3. 提取基础特征并应用 Log 压缩
    # 传入 [1, time] 形状，提取器内部 unsqueeze(1) 后变为 [1, 1, time]，顺利 pad 时间轴
    linear_mel = mel_extractor(wav) 
    
    # 参考官方压缩逻辑，直接压缩并截断
    base_log_mel = dynamic_range_compression_torch(linear_mel, clip_val=1e-9)

    print(f"Successfully extracted Log-Mel with shape: {base_log_mel.shape}")

    # ==========================================
    # 4. 运行测试并生成四张可视化图片
    # ==========================================

    # Test 1: 物理底噪添加
    mel_noise = add_diverse_background_noise(base_log_mel, noise_level=0.001)
    plot_and_save_comparison(
        base_log_mel, mel_noise, 
        title="Physical Background Noise", 
        save_path=os.path.join(output_dir, "aug_1_background_noise.png")
    )

    # Test 2: 频率掩蔽 (Frequency Masking)
    mel_f_mask = frequency_masking(base_log_mel, freq_mask_param=3, num_masks=2)
    plot_and_save_comparison(
        base_log_mel, mel_f_mask, 
        title="Frequency Masking", 
        save_path=os.path.join(output_dir, "aug_2_frequency_mask.png")
    )

    # Test 3: 时间掩蔽 (Time Masking)
    mel_t_mask = time_masking(base_log_mel, time_mask_param=3, num_masks=2)
    plot_and_save_comparison(
        base_log_mel, mel_t_mask, 
        title="Time Masking", 
        save_path=os.path.join(output_dir, "aug_3_time_mask.png")
    )

    # Test 4: 局部时频噪声块 (Cutout/Interference)
    mel_block_mask = time_frequency_noise_block(
        base_log_mel, 
        num_blocks=5, 
        max_time_ratio=0.2, 
        max_freq_ratio=0.15, 
        noise_mode='replace'
    )
    plot_and_save_comparison(
        base_log_mel, mel_block_mask, 
        title="Time-Frequency Noise Block (Replace)", 
        save_path=os.path.join(output_dir, "aug_4_tf_block.png")
    )
    
    print("\nAll unit tests passed and images generated successfully.")

if __name__ == "__main__":
    # 运行测试，你需要替换为一个真实的测试音频文件路径
    TEST_WAV_PATH = "test_audio.wav" 
    
    # 如果本地没有测试文件，可以临时生成一个白噪声信号来防止报错（实际使用时请替换）
    if not os.path.exists(TEST_WAV_PATH):
        print("Warning: test_audio.wav not found. Generating a 3-second sweep signal for testing...")
        sample_rate = 16000
        t = torch.linspace(0, 3, 3 * sample_rate)
        # 生成一个包含频率变化的扫频信号以更好地观察谱图
        dummy_wav = torch.sin(2 * 3.14159 * 440 * (t + t**2 / 2)).unsqueeze(0)
        torchaudio.save(TEST_WAV_PATH, dummy_wav, sample_rate)

    test_data_augmentations(wav_path=TEST_WAV_PATH)