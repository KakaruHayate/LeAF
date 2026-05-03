import torch
import torch.nn as nn
import torch.nn.functional as F


class LeAF(nn.Module):
    """
    Latent Engine A × Five (LeAF) — JEPA‑style world model adapted for learning
    disentangled acoustic curves from Mel spectrograms with optional distillation.
    """

    def __init__(
        self,
        encoder,          # MelViTModel
        predictor,        # ARPredictor
        action_encoder,   # Embedder
        decoder=None,     # Decoder
        projector=None,   # MLP applied to encoder output
        pred_proj=None,   # MLP applied to predictor output
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.decoder = decoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """
        Encode mel spectrograms and optional actions.

        Args:
            info (dict):
                'mel'    : (B, C, n_mels, T)   float tensor
                'action' : (B, T) or (B, T, 1) float tensor

        Returns:
            info (dict): updated with:
                'emb'     : (B, T, D)        projected frame‑level latent embeddings
                'act_emb' : (B, T, A_emb)    action embeddings (if action present)
        """
        mel = info['mel'].float()                     # (B, C, n_mels, T)
        B, C, n_mels, T = mel.shape

        seq_out = self.encoder(mel)                  # (B, 1+T, hidden)
        frame_tokens = seq_out[:, 1:, :]             # discard CLS -> (B, T, hidden)

        # Per‑frame projection
        frame_flat = frame_tokens.reshape(B * T, -1) # (B*T, hidden)
        emb = self.projector(frame_flat)             # (B*T, D)
        emb = emb.view(B, T, -1)                     # (B, T, D)
        info['emb'] = emb

        # Action encoding
        if 'action' in info:
            act = info['action'].float()
            if act.dim() == 2:
                act = act.unsqueeze(-1)              # (B, T, 1)
            act_emb = self.action_encoder(act)       # (B, T, A_emb)
            info['act_emb'] = act_emb

        return info

    def predict(self, emb, act_emb):
        """
        Autoregressive next‑step prediction.

        Args:
            emb     : (B, T, D)       frame latents
            act_emb : (B, T, A_emb)   action embeddings

        Returns:
            preds   : (B, T, D')      predicted next‑step latents
        """
        B, T, D = emb.shape
        preds = self.predictor(emb, act_emb)         # (B, T, D) or (B, T, D')
        preds_flat = preds.reshape(B * T, -1)        # (B*T, D')
        preds = self.pred_proj(preds_flat)           # (B*T, D')
        preds = preds.view(B, T, -1)                 # (B, T, D')
        return preds

    def forward(self, info, mode='train'):
        """
        Forward pass with training / inference branching.

        Args:
            info (dict):  'mel' and (for training) 'action'.
            mode (str):   'train' or 'infer'.

        Returns (train):
            if self.decoder is not None:
                emb, preds, curve_pred   (curve_pred: (B, T, curve_dim) normalised)
            else:
                emb, preds               (world‑model‑only phase)

        Returns (infer):
            curve (B, T, curve_dim)      denormalised output (requires decoder)
        """
        info = self.encode(info)

        if mode == 'infer':
            if self.decoder is None:
                raise RuntimeError("Decoder is required for inference, got None.")
            emb = info['emb']                     # (B, T, D)
            curve = self.decoder.infer(emb)       # (B, T, curve_dim)
            return curve

        # ---------- training ----------
        emb = info['emb']                         # (B, T, D)
        act_emb = info['act_emb']                 # (B, T, A_emb)
        preds = self.predict(emb, act_emb)        # (B, T, D')

        if self.decoder is not None:
            curve_pred = self.decoder(emb)        # (B, T, curve_dim)
            return emb, preds, curve_pred
        else:
            return emb, preds
