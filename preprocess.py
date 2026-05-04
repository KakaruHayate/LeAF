import json
import pathlib
import sys
import os
import click
import librosa
import numpy as np
import torch
import tqdm
import pandas
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
from r3moe.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch
from r3moe.nets import BiLSTMCurveEstimator

# 1. 直接从 get_pitch 导入工具函数，保持代码 DRY (Don't Repeat Yourself)
from get_pitch import interp_f0, resample_align_curve
from rmvpe import RMVPE

# ---------------------------------------------------------
# 核心工具类与函数
# ---------------------------------------------------------

def get_pitch_rmvpe_fast(wav_data, hop_size, audio_sample_rate, rmvpe_model, target_length, interp_uv=True):
    """复用 get_pitch 算法，传入全局加载的模型，对齐到 Mel 长度"""
    f0 = rmvpe_model.infer_from_audio(wav_data, sample_rate=audio_sample_rate)
    uv = f0 == 0
    f0, uv = interp_f0(f0, uv)
    
    time_step = hop_size / audio_sample_rate
    f0_res = resample_align_curve(f0, 0.01, time_step, target_length)
    uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, target_length) > 0.5
    
    if not interp_uv:
        f0_res[uv_res] = 0
    return f0_res, uv_res


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect
    sig = inspect.signature(kwarg_obj)
    filter_keys = [
        param.name for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    return {k: dict_to_filter[k] for k in filter_keys if k in dict_to_filter}


class FastCurveEstimator:
    """提取自 eval.py，用于批处理推理，输出作为感知滑块控制的 OPEC 曲线"""
    def __init__(self, model_path: pathlib.Path, device: str):
        config_path = model_path.with_name("config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.device = device
        self.model = BiLSTMCurveEstimator(**filter_kwargs(config["model_args"], BiLSTMCurveEstimator))
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict({k: v for k, v in state_dict.items() if 'k_filter' not in k}, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def estimate_from_mel(self, mel: np.ndarray, length: int) -> np.ndarray:
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(self.device)
        pred_curve = self.model(mel_tensor).squeeze(0).cpu().numpy().squeeze()
        
        if len(pred_curve) != length:
            pred_curve = np.interp(
                np.linspace(0, len(pred_curve) - 1, length),
                np.arange(len(pred_curve)),
                pred_curve
            ).astype(np.float32)
        return pred_curve


# ---------------------------------------------------------
# 多线程 Worker 函数
# ---------------------------------------------------------

def process_single(audio_file, args):
    """单线程处理逻辑，无状态，通过 shared_args 传递模型引用"""
    source_dir = args["source_dir"]
    target_dir = args["target_dir"]
    sample_rate = args["sample_rate"]
    hop_size = args["hop_size"]
    mel_spec_transform = args["mel_spec_transform"]
    rmvpe_model = args["rmvpe_model"]
    opec_estimator = args["opec_estimator"]

    try:
        target_file = target_dir / audio_file.relative_to(source_dir).with_suffix(".npz")
        mel, has_existing_mel = None, False

        # 需求 5: 存在 npz 时，检查并复用 Mel
        if target_file.exists():
            data = np.load(target_file)
            if 'spectrogram' in data:
                mel = data['spectrogram']
                has_existing_mel = True
                # 如果已经完全处理过，直接跳过
                if 'pitch' in data and 'opec' in data:
                    return True, mel.shape[0], target_file.relative_to(target_dir).as_posix(), None

        # 读取音频
        audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

        # 需求 1: 如果没有现成的 Mel，则提取
        if not has_existing_mel:
            with torch.no_grad():
                mel = dynamic_range_compression_torch(
                    mel_spec_transform(torch.from_numpy(audio)[None]), clip_val=1e-9
                )[0].T.cpu().numpy()
            
        mel_length = mel.shape[0]

        # 需求 2: 使用 RMVPE 提取 Pitch 并插值
        f0_res, uv_res = get_pitch_rmvpe_fast(audio, hop_size, sample_rate, rmvpe_model, mel_length)
        
        # 需求 4: 如果全是 UV (静音/无声)，则跳过该片段
        if np.all(uv_res):
            return False, None, None, "skip: 全片段为 UV (静音/无调性)"

        # 需求 3: 使用 R3MOE 提取 OPEC 曲线
        opec_res = opec_estimator.estimate_from_mel(mel, mel_length)

        # 保存打包
        target_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target_file, spectrogram=mel, pitch=f0_res, opec=opec_res)

        return True, mel_length, target_file.relative_to(target_dir).as_posix(), None

    except Exception as e:
        return False, None, None, f"error: {str(e)}"


# ---------------------------------------------------------
# 主入口
# ---------------------------------------------------------

@click.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("target_dir", type=click.Path(path_type=pathlib.Path))
@click.option("--rmvpe_model", type=click.Path(exists=True, path_type=pathlib.Path), required=True, help="Path to rmvpe.pt")
@click.option("--opec_model", type=click.Path(exists=True, path_type=pathlib.Path), required=True, help="Path to R3MOE curve estimator .pth")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--num_workers", default=None, type=int, help="线程数，默认根据 CPU 核心数自动决定")
@click.option("--val_num", default=8, type=int)
@click.option("--sample_rate", default=16000, type=int)
@click.option("--mel_bins", default=80, type=int)
@click.option("--hop_size", default=320, type=int)
@click.option("--win_size", default=1024, type=int)
@click.option("--f_min", default=0, type=int)
@click.option("--f_max", default=None, type=int)
def preprocess(source_dir, target_dir, rmvpe_model, opec_model, device, num_workers, val_num, sample_rate, mel_bins, hop_size, win_size, f_min, f_max):
    audio_list = sorted(f for ext in ["*.m4a", "*.wav", "*.mp3", "*.aac"] for f in source_dir.rglob(ext))
    if not audio_list:
        print("未找到音频文件。")
        return
        
    target_dir.mkdir(parents=True, exist_ok=True)

    # 预加载模型和组件到设备中（只加载一次）
    shared_args = {
        "source_dir": source_dir,
        "target_dir": target_dir,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "mel_spec_transform": PitchAdjustableMelSpectrogram(
            sample_rate=sample_rate, n_fft=win_size, win_length=win_size, 
            hop_length=hop_size, f_min=f_min, f_max=f_max, n_mels=mel_bins, center=True
        ),
        "rmvpe_model": RMVPE(rmvpe_model),
        "opec_estimator": FastCurveEstimator(opec_model, device)
    }

    # 多线程参数配置 (使用 CUDA 时注意显存，不要开太大)
    max_workers = num_workers if num_workers else min(os.cpu_count() or 4, 8)
    print(f"开始多线程处理，启动 {max_workers} 个线程在 {device} 上运行...")

    results = {}
    messages = []
    
    # 启用线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single, f, shared_args): f for f in audio_list}
        with tqdm.tqdm(total=len(audio_list), desc="数据预处理进度") as pbar:
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    success, length, npz_path, msg = future.result()
                    if success:
                        results[f] = (length, npz_path)
                    if msg:
                        messages.append((f.relative_to(source_dir).as_posix(), msg))
                except Exception as e:
                    messages.append((f.relative_to(source_dir).as_posix(), f"Unhandled error: {str(e)}"))
                pbar.update(1)

    # 导出跳过/报错日志
    if messages:
        df_msg = pandas.DataFrame(messages, columns=["file", "message"])
        df_msg.to_csv(target_dir / "skips_and_errors.csv", index=False, encoding="utf-8-sig")
        print(f"\n保存了 {len(messages)} 条跳过或错误信息到 skips_and_errors.csv")

    # 提取结果并排序以保证稳定性
    sorted_files = sorted(results.keys())
    len_list = [results[f][0] for f in sorted_files]
    npz_list = [results[f][1] for f in sorted_files]
    
    if not npz_list:
        print("没有成功处理任何文件，退出。")
        return

    # 生成训练集与验证集切分
    val_indices = sorted(np.random.choice(len(len_list), min(val_num, len(len_list)), replace=False))
    
    with open(target_dir / "train.txt", "w", encoding="utf8") as f:
        f.write("\n".join(npz_list) + "\n")
    with open(target_dir / "valid.txt", "w", encoding="utf8") as f:
        f.write("\n".join([npz_list[i] for i in val_indices]) + "\n")
        
    np.save(target_dir / "lengths.npy", len_list)
    with open(target_dir / "metadata.json", "w", encoding="utf8") as f:
        json.dump({
            "sample_rate": sample_rate, 
            "mel_bins": mel_bins,
            "hop_size": hop_size,
            "win_size": win_size,
            "f_min": f_min,
            "f_max": f_max
        }, f, indent=2)
        
    print(f"\n处理完成！共成功预处理 {len(npz_list)} 个文件。")

if __name__ == "__main__":
    preprocess()