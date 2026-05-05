import torch


def add_diverse_background_noise(mel: torch.Tensor, noise_level: float = 0.001, 
                                 noise_type: str = 'random', eps: float = 1e-5) -> torch.Tensor:
    """
    Simulates various types of physical background noise by applying spectral tilts 
    in the linear Mel-frequency domain.
    
    Args:
        mel (Tensor): Log-Mel spectrogram, shape (..., n_mels, time).
        noise_level (float): The base amplitude of the injected noise.
        noise_type (str): 'white', 'pink', 'brown', 'bandpass', or 'random'.
        eps (float): Minimum energy floor for clamping.
        
    Returns:
        Tensor: Augmented Log-Mel spectrogram.
    """
    # 1. To Linear Domain
    linear_mel = torch.exp(mel).clone()
    n_mels = linear_mel.shape[-2]
    
    # 2. Generate base non-negative physical noise
    noise = torch.abs(torch.randn_like(linear_mel))
    
    # 3. Determine noise type
    types = ['white', 'pink', 'brown', 'bandpass']
    current_type = noise_type
    if current_type == 'random':
        idx = torch.randint(0, len(types), (1,)).item()
        current_type = types[idx]
        
    # 4. Apply Spectral Tilt (Coloring the noise)
    if current_type == 'pink':
        # Pink Noise: Energy decays as 1/sqrt(f)
        # We create a column vector to scale the mel frequency bins
        freq_idx = torch.arange(1, n_mels + 1, device=mel.device).view(-1, 1).float()
        tilt = 1.0 / torch.sqrt(freq_idx)
        noise = noise * tilt
        
    elif current_type == 'brown':
        # Brown Noise: Energy decays steeply as 1/f (heavy low-frequency rumble)
        freq_idx = torch.arange(1, n_mels + 1, device=mel.device).view(-1, 1).float()
        tilt = 1.0 / freq_idx
        noise = noise * tilt
        
    elif current_type == 'bandpass':
        # Bandpass Noise: Interference concentrated in a random specific frequency band
        band_width = torch.randint(n_mels // 8, n_mels // 3 + 1, (1,)).item()
        start_f = torch.randint(0, n_mels - band_width + 1, (1,)).item()
        
        mask = torch.zeros_like(noise)
        mask[..., start_f:start_f + band_width, :] = 1.0
        # Boost local noise slightly since the rest of the spectrum is quiet
        noise = noise * mask * 1.5 
        
    # (White noise requires no modification, as randn is already white)

    # 5. Scale by global noise level and add
    linear_mel = linear_mel + (noise * noise_level)
    
    # 6. Back to Log Domain
    return torch.log(torch.clamp(linear_mel, min=eps))


def add_physical_background_noise(mel: torch.Tensor, noise_level: float = 0.01, eps: float = 1e-5) -> torch.Tensor:
    """
    Simulates physical background noise. 
    Converts Log-Mel to linear energy, adds non-negative noise, and reverts to Log-Mel.
    
    Args:
        mel (Tensor): Log-Mel spectrogram, shape (..., n_mels, time).
        noise_level (float): The amplitude of the injected noise.
        eps (float): Minimum energy floor for clamping.
        
    Returns:
        Tensor: Augmented Log-Mel spectrogram.
    """
    # 1. To Linear Domain
    linear_mel = torch.exp(mel)
    
    # 2. Add physical noise (absolute value ensures positive energy addition)
    noise = torch.abs(torch.randn_like(linear_mel)) * noise_level 
    linear_mel = linear_mel + noise
    
    # 3. Back to Log Domain
    return torch.log(torch.clamp(linear_mel, min=eps))


def frequency_masking(mel: torch.Tensor, freq_mask_param: int = 27, num_masks: int = 1, eps: float = 1e-5) -> torch.Tensor:
    """
    Frequency Masking: Randomly masks consecutive frequency channels in the linear domain.
    
    Reference:
        Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method 
        for Automatic Speech Recognition." Interspeech 2019.
        
    Args:
        mel (Tensor): Log-Mel spectrogram, shape (..., n_mels, time).
        freq_mask_param (int): Maximum width of the frequency mask (F).
        num_masks (int): Number of frequency masks to apply (m_F).
        eps (float): Minimum energy floor used to represent "silence/erasure".
        
    Returns:
        Tensor: Augmented Log-Mel spectrogram.
    """
    # Clone to avoid in-place modification issues across batches
    linear_mel = torch.exp(mel).clone()
    n_mels = linear_mel.shape[-2]
    
    for _ in range(num_masks):
        # Generate random length and start point
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, n_mels - f + 1, (1,)).item()
        
        # Erase in linear domain using the physical noise floor (eps)
        linear_mel[..., f0:f0 + f, :] = eps
        
    return torch.log(torch.clamp(linear_mel, min=eps))


def time_masking(mel: torch.Tensor, time_mask_param: int = 100, num_masks: int = 1, eps: float = 1e-5) -> torch.Tensor:
    """
    Time Masking: Randomly masks consecutive time frames in the linear domain.
    
    Reference:
        Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method 
        for Automatic Speech Recognition." Interspeech 2019.
        
    Args:
        mel (Tensor): Log-Mel spectrogram, shape (..., n_mels, time).
        time_mask_param (int): Maximum length of the time mask (T).
        num_masks (int): Number of time masks to apply (m_T).
        eps (float): Minimum energy floor used to represent "silence/erasure".
        
    Returns:
        Tensor: Augmented Log-Mel spectrogram.
    """
    linear_mel = torch.exp(mel).clone()
    n_frames = linear_mel.shape[-1]
    
    for _ in range(num_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, n_frames - t + 1, (1,)).item()
        
        # Erase in linear domain
        linear_mel[..., :, t0:t0 + t] = eps
        
    return torch.log(torch.clamp(linear_mel, min=eps))


def time_frequency_noise_block(mel: torch.Tensor, num_blocks: int = 5, 
                               max_time_ratio: float = 0.3, max_freq_ratio: float = 0.05, 
                               noise_mode: str = 'add', noise_std: float = 0.1, 
                               eps: float = 1e-5) -> torch.Tensor:
    """
    Places random rectangular noise blocks on the spectrogram in the linear domain. 
    Can simulate transient interference (additive noise) or localized 
    information loss (erasing).
    
    Reference:
        "FCPE: A Fast Context-based Pitch Estimation Model" (arXiv: 2509.15140)
    
    Args:
        mel (Tensor): Log-Mel spectrogram, shape (..., n_mels, time).
        num_blocks (int): Number of rectangular blocks to apply.
        max_time_ratio (float): Max temporal width relative to total frames.
        max_freq_ratio (float): Max frequency height relative to total mels.
        noise_mode (str): 'add' -> Superimposes Gaussian noise (interference).
                          'replace' -> Erases the region with `eps` (loss).
        noise_std (float): Standard deviation of the additive noise.
        eps (float): Minimum energy floor for clamping and replacement.
        
    Returns:
        Tensor: Augmented Log-Mel spectrogram.
    """
    linear_mel = torch.exp(mel).clone()
    n_mels, n_frames = linear_mel.shape[-2], linear_mel.shape[-1]
    
    for _ in range(num_blocks):
        # Calculate random block dimensions
        block_freq = max(1, int(max_freq_ratio * n_mels * torch.rand(1).item()))
        block_time = max(1, int(max_time_ratio * n_frames * torch.rand(1).item()))

        # Calculate random start positions
        f0 = torch.randint(0, n_mels - block_freq + 1, (1,)).item()
        t0 = torch.randint(0, n_frames - block_time + 1, (1,)).item()

        if noise_mode == 'add':
            # Additive interference in linear energy domain
            # We scale the existing region or simply add fresh absolute noise
            noise = torch.abs(torch.randn_like(linear_mel[..., f0:f0+block_freq, t0:t0+block_time])) * noise_std
            linear_mel[..., f0:f0+block_freq, t0:t0+block_time] += noise
            
        elif noise_mode == 'replace':
            # Complete erasure of the localized block in linear domain
            linear_mel[..., f0:f0+block_freq, t0:t0+block_time] = eps
            
    return torch.log(torch.clamp(linear_mel, min=eps))