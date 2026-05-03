import numpy as np
from rmvpe import RMVPE


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


def get_pitch_rmvpe(wav_data, hop_size, audio_sample_rate, interp_uv=True):
    rmvpe = RMVPE(pathlib.Path(__file__).parent / 'pretrain' / 'rmvpe' / 'model.pt')
    f0 = rmvpe.infer_from_audio(wav_data, sample_rate=audio_sample_rate)
    uv = f0 == 0
    f0, uv = interp_f0(f0, uv)

    time_step = hop_size / audio_sample_rate
    length = (wav_data.shape[0] + hop_size - 1) // hop_size
    f0_res = resample_align_curve(f0, 0.01, time_step, length)
    uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
    if not interp_uv:
        f0_res[uv_res] = 0
    return time_step, f0_res, uv_res
