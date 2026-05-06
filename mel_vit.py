import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from muon_layer import AdamWLinear, AdamWCov1d, AdamWCov2d


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MelPatchEmbeddings(nn.Module):
    """
    Converts a Mel-spectrogram of shape `(batch, in_channels, n_mels, time_frames)` into 
    patch embeddings. By setting kernel_size=(n_mels, 1), each time frame becomes a single token.
    """
    def __init__(self, n_mels: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.n_mels = n_mels
        # Convolution over the entire frequency axis, sliding 1 frame at a time.
        self.projection = AdamWCov2d(
            in_channels, 
            hidden_size, 
            kernel_size=(n_mels, 1), 
            stride=(n_mels, 1)
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        b, c, m, t = mel_spectrogram.shape
        if m != self.n_mels:
            raise ValueError(f"Input mel bins ({m}) does not match model config ({self.n_mels})")
            
        x = self.projection(mel_spectrogram)  # (batch, hidden, 1, time_frames)
        x = x.flatten(2).transpose(1, 2)      # (batch, time_frames, hidden)
        return x


class MelViTEmbeddings(nn.Module):
    """
    Constructs the embeddings from patch embeddings, appends the CLS token, 
    and adds dynamic 1D sinusoidal positional embeddings.
    """
    def __init__(self, n_mels: int, in_channels: int, hidden_size: int, dropout: float, max_len: int = 8192):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embeddings = MelPatchEmbeddings(n_mels, in_channels, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute sinusoidal positional embeddings
        pe = self._build_sincos_position_embedding(max_len, hidden_size)
        self.register_buffer('position_embeddings', pe, persistent=False)

    def _build_sincos_position_embedding(self, max_len: int, hidden_size: int) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        batch = mel_spectrogram.shape[0]
        embeddings = self.patch_embeddings(mel_spectrogram)
        seq_length = embeddings.shape[1]
        
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # total sequence length = frames + 1 (for CLS)
        embeddings = embeddings + self.position_embeddings[:, :seq_length + 1, :]
        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool, dropout: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.dropout = dropout
        self.scaling = self.attention_head_size ** -0.5
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        new_shape = (batch, seq_len, self.num_attention_heads, self.attention_head_size)
        
        q = self.query(hidden_states).view(new_shape).transpose(1, 2)
        k = self.key(hidden_states).view(new_shape).transpose(1, 2)
        v = self.value(hidden_states).view(new_shape).transpose(1, 2)
        
        context, attn_weights = eager_attention_forward(
            self, q, k, v, scaling=self.scaling, dropout=self.dropout if self.training else 0.0,
        )
        context = context.reshape(batch, -1, self.all_head_size)
        return context, attn_weights


class ViTSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool, attention_dropout: float, output_dropout: float):
        super().__init__()
        self.attention = ViTSelfAttention(hidden_size, num_attention_heads, qkv_bias, attention_dropout)
        self.output = ViTSelfOutput(hidden_size, output_dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_output, _ = self.attention(hidden_states)
        return self.output(self_output)


class ViTIntermediate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.dense = AdamWLinear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + input_tensor


class ViTLayer(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool, attention_dropout: float, hidden_dropout: float, layer_norm_eps: float):
        super().__init__()
        self.attention = ViTAttention(hidden_size, num_attention_heads, qkv_bias, attention_dropout, hidden_dropout)
        self.intermediate = ViTIntermediate(hidden_size)
        self.output = ViTOutput(hidden_size, hidden_dropout)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layernorm_before(hidden_states)
        x = self.attention(x)
        hidden_states = hidden_states + x
        x = self.layernorm_after(hidden_states)
        x = self.intermediate(x)
        x = self.output(x, hidden_states)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(**layer_kwargs) for _ in range(num_layers)])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class MelViTModel(nn.Module):
    """
    Vision Transformer tailored for Mel-spectrograms in JEPA architectures.
    Each frame acts as a single token. Automatically pads temporal dimension 
    to a specified multiple before encoding, and slices it back afterwards.
    """
    def __init__(
        self,
        n_mels: int = 80,
        in_channels: int = 1,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        pad_to_multiple: int = 80,  # Pad frames to a multiple of this value
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.pad_to_multiple = pad_to_multiple
        
        self.embeddings = MelViTEmbeddings(
            n_mels=n_mels, 
            in_channels=in_channels, 
            hidden_size=hidden_size, 
            dropout=hidden_dropout_prob,
            max_len=max_position_embeddings
        )
        layer_kwargs = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "qkv_bias": qkv_bias,
            "attention_dropout": attention_probs_dropout_prob,
            "hidden_dropout": hidden_dropout_prob,
            "layer_norm_eps": layer_norm_eps,
        }
        self.encoder = ViTEncoder(num_hidden_layers, **layer_kwargs)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Apply standard ViT initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, MelViTEmbeddings):
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=self.initializer_range)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: Tensor of shape (batch, in_channels, n_mels, time_frames)
        """
        b, c, m, orig_time_frames = mel_spectrogram.shape
        
        # 1. Dynamic Padding
        pad_len = (self.pad_to_multiple - (orig_time_frames % self.pad_to_multiple)) % self.pad_to_multiple
        if pad_len > 0:
            mel_spectrogram = F.pad(mel_spectrogram, (0, pad_len))
            
        # 2. Extract Embeddings (Tokens)
        embedding_output = self.embeddings(mel_spectrogram)
        
        # 3. Transformer Encoder
        encoder_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_output)
        
        # 4. Slice back to original length
        # sequence_output shape: (B, 1 + padded_frames, hidden)
        if pad_len > 0:
            cls_token = sequence_output[:, 0:1, :]
            valid_frames = sequence_output[:, 1: 1 + orig_time_frames, :]
            sequence_output = torch.cat([cls_token, valid_frames], dim=1)
            
        return sequence_output
