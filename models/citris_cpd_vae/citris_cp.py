"""
CITRIS + TS-CP^2 VAE

- Uses CITRISVAE encoder/decoder as base.
- Adds:
  * Temporal window encoder f_psi over z_t
  * TS-CP^2-style CPC loss on history/future windows
  * Change scores c_t from history/future similarity drop
  * Soft intervention gates g_t(z_{t-1}, z_t, c_t)
  * Gated Gaussian transition prior p(z_{t+1} | z_t, g_{t+1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 프로젝트 구조에 맞게 import 경로를 조정해줘.
# lightning_module.py 안에 CITRISVAE가 정의되어 있다고 가정.
from models.citris_vae.lightning_module import CITRISVAE
from models.shared import CosineWarmupScheduler


class TemporalWindowEncoder(nn.Module):
    """
    f_psi: maps a temporal window of z's to a compact embedding.
    Input : windows of shape (B, N, L, D)
    Output: embeddings of shape (B, N, H)
    """

    def __init__(self, latent_dim: int, window_len: int, hidden_dim: int):
        super().__init__()
        self.window_len = window_len
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim * window_len, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        windows: (B, N, L, D)
        returns: (B, N, H), L2-normalized for cosine similarity
        """
        B, N, L, D = windows.shape
        assert L == self.window_len and D == self.latent_dim

        x = windows.reshape(B * N, L * D)
        h = self.net(x)
        h = F.normalize(h, dim=-1)
        return h.view(B, N, -1)


class CITRISTSCP2VAE(CITRISVAE):
    """
    CITRIS-VAE with TS-CP^2-style change-point module and
    soft intervention gates g_t that replace explicit I^{t+1}.

    Changes compared to CITRISVAE:
    - No explicit intervention labels I^{t+1}
    - Temporal window encoder over z_t (TS-CP^2)
    - CPC loss on history/future window embeddings
    - Change scores c_t from similarity drops
    - Gated Gaussian transition prior controlled by g_t
    - Additional gate sparsity regularizer
    """

    def __init__(
        self,
        *args,
        window_len: int = 4,
        ma_window: int = 4,
        cpc_dim: int = 128,
        cpc_tau: float = 0.1,
        lambda_cpc: float = 1.0,
        lambda_gate: float = 1e-3,
        **kwargs,
    ):
        """
        Additional hyperparameters
        --------------------------
        window_len : int
            L, history/future window length.
        ma_window : int
            Moving-average window length for similarity smoothing.
        cpc_dim : int
            Dimensionality of h^H_t, h^F_t embeddings.
        cpc_tau : float
            InfoNCE temperature.
        lambda_cpc : float
            Weight for CPC loss.
        lambda_gate : float
            Weight for gate sparsity loss.
        """
        super().__init__(*args, **kwargs)

        self.window_len = window_len
        self.ma_window = ma_window
        self.cpc_tau = cpc_tau
        self.lambda_cpc = lambda_cpc
        self.lambda_gate = lambda_gate

        # latent dim (total) and num causal vars from CITRIS hparams
        D = self.hparams.num_latents
        K = self.hparams.num_causal_vars

        # --- TS-CP^2 temporal window encoder f_psi ---
        self.window_encoder = TemporalWindowEncoder(
            latent_dim=D,
            window_len=self.window_len,
            hidden_dim=cpc_dim,
        )

        # --- Soft intervention gate MLP ---
        # Input = [z_t, z_{t+1}, c_{t+1}] -> dim 2D + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * D + 1, self.hparams.c_hid),
            nn.SiLU(),
            nn.Linear(self.hparams.c_hid, K),
        )

        # --- Gated Gaussian transition prior parameters ---
        # dynamics prior p_dyn(z_{t+1} | z_t)
        self.dyn_mu = nn.Linear(D, D)
        self.dyn_logvar = nn.Linear(D, D)

        # intervention prior p_int(z_{t+1})
        self.mu_int = nn.Parameter(torch.zeros(1, D))
        self.logvar_int = nn.Parameter(torch.zeros(1, D))

        # map latent dims to causal blocks (simple contiguous blocks)
        block_size = max(D // K, 1)
        assign = torch.arange(D) // block_size  # (D,)
        assign = torch.clamp(assign, max=K - 1)
        self.register_buffer("latent_to_block", assign.long())

    # ------------------------------------------------------------------
    # Optimizer: keep it simple, single AdamW over all parameters
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.0,
        )
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            max_iters=self.hparams.max_iters,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    # ------------------------------------------------------------------
    # TS-CP^2 helpers
    # ------------------------------------------------------------------
    def _moving_average(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, N)  -> moving average over N with window ma_window.
        """
        B, N = s.shape
        if self.ma_window <= 1 or N == 0:
            return s

        w = self.ma_window
        # pad on the left so output length stays N
        s_pad = F.pad(s, (w - 1, 0), mode="replicate")  # (B, N + w - 1)
        kernel = s.new_ones(1, 1, w) / w
        ma = F.conv1d(s_pad.unsqueeze(1), kernel).squeeze(1)  # (B, N)
        return ma

    def _cpc_and_change_scores(self, z_mean: torch.Tensor):
        """
        z_mean: (B, T, D)
        Returns:
            c: (B, T-1) change scores per transition t -> t+1
            cpc_loss: scalar
        """
        B, T, D = z_mean.shape
        L = self.window_len

        # ✅ 시퀀스가 너무 짧으면 CPC/변화점 신호를 계산하지 않음
        # window_encoder는 고정 길이 MLP이므로 L을 임의로 바꾸면 안 됨
        if T < 2 * L + 1:
            c = z_mean.new_zeros(B, T - 1)
            cpc_loss = z_mean.new_zeros(())
            return c, cpc_loss

        centers = list(range(L - 1, T - L))
        if len(centers) == 0:
            c = z_mean.new_zeros(B, T - 1)
            cpc_loss = z_mean.new_zeros(())
            return c, cpc_loss


        hist_windows = []
        fut_windows = []
        for t in centers:
            # history: z_{t-L+1 : t}
            hist_windows.append(z_mean[:, t - L + 1 : t + 1, :])
            # future: z_{t+1 : t+L+1}
            fut_windows.append(z_mean[:, t + 1 : t + 1 + L, :])

        # (B, N, L, D)
        hist_windows = torch.stack(hist_windows, dim=1)
        fut_windows = torch.stack(fut_windows, dim=1)
        B, N, _, _ = hist_windows.shape

        # encode windows -> (B, N, H)
        h_H = self.window_encoder(hist_windows)
        h_F = self.window_encoder(fut_windows)

        # --- InfoNCE CPC loss ---
        H = h_H.shape[-1]
        anchor = h_H.reshape(B * N, H)  # positives: same index
        pos = h_F.reshape(B * N, H)

        # cosine similarities via dot product (already normalized)
        logits = torch.matmul(anchor, pos.T) / self.cpc_tau  # (B*N, B*N)
        labels = torch.arange(B * N, device=z_mean.device)
        cpc_loss = F.cross_entropy(logits, labels)

        # --- change scores c_t ---
        # s_t = cos(h^H_t, h^F_t)
        s = (h_H * h_F).sum(dim=-1)  # (B, N)
        s_ma = self._moving_average(s)
        c_cent = s_ma - s  # (B, N)

        # map from centers (N) to all transitions (T-1)
        c = z_mean.new_zeros(B, T - 1)
        for idx, t in enumerate(centers):
            # transition t -> t+1
            c[:, t] = c_cent[:, idx]

        return c, cpc_loss

    def _compute_gates(self, z_mean: torch.Tensor, c: torch.Tensor):
        """
        z_mean: (B, T, D)
        c     : (B, T-1)  change scores per transition
        returns:
            g: (B, T-1, K) soft intervention gates
        """
        B, T, D = z_mean.shape
        K = self.hparams.num_causal_vars

        z_t = z_mean[:, :-1, :]   # (B, T-1, D)
        z_tp1 = z_mean[:, 1:, :]  # (B, T-1, D)

        feat = torch.cat(
            [z_t, z_tp1, c.unsqueeze(-1)], dim=-1
        )  # (B, T-1, 2D + 1)

        g_logits = self.gate_mlp(feat)  # (B, T-1, K)
        g = torch.sigmoid(g_logits)
        return g

    def _gated_prior_kl(
        self,
        z_mean: torch.Tensor,
        z_logstd: torch.Tensor,
        g: torch.Tensor,
    ):
        """
        z_mean   : (B, T, D)
        z_logstd : (B, T, D)  log sigma of q(z_t | x_t)
        g        : (B, T-1, K) soft gates per causal factor

        returns:
            kl_per_seq: (B,)  summed over time and latent dims
        """
        B, T, D = z_mean.shape
        K = self.hparams.num_causal_vars

        z_t = z_mean[:, :-1, :]      # (B, T-1, D)
        z_tp1 = z_mean[:, 1:, :]     # (B, T-1, D)
        q_logvar = 2.0 * z_logstd[:, 1:, :]  # log sigma^2

        # dynamics prior parameters
        mu_dyn = self.dyn_mu(z_t)            # (B, T-1, D)
        logvar_dyn = self.dyn_logvar(z_t)    # (B, T-1, D)
        var_dyn = torch.exp(logvar_dyn)

        # intervention prior parameters (broadcast)
        mu_int = self.mu_int        # (1, D)
        var_int = torch.exp(self.logvar_int)  # (1, D)

        # map causal gates g_t(i) to latent dims (blocks)
        # latent_to_block: (D,), values in [0, K-1]
        idx = self.latent_to_block.view(1, 1, D).expand(B, T - 1, D)
        g_latent = g.gather(-1, idx)          # (B, T-1, D)

        # convex combination of dyn/int priors
        var_prior = (1.0 - g_latent) * var_dyn + g_latent * var_int
        mu_prior = (1.0 - g_latent) * mu_dyn + g_latent * mu_int
        logvar_prior = torch.log(var_prior + 1e-8)

        # diagonal Gaussian KL(q || p)
        kl = 0.5 * (
            logvar_prior
            - q_logvar
            + (torch.exp(q_logvar) + (z_tp1 - mu_prior) ** 2) / (var_prior + 1e-8)
            - 1.0
        )  # (B, T-1, D)

        kl = kl.sum(dim=-1).sum(dim=-1)  # sum over D and time -> (B,)
        return kl

    # ------------------------------------------------------------------
    # Main training loss (based on CITRISVAE _get_loss, but:
    # - no supervised I^{t}
    # - adds CPC + gate losses
    # - uses gated transition prior
    # ------------------------------------------------------------------
    def _get_loss(self, batch, mode: str = "train"):
        """
            batch:
            - (imgs, target)
            - (imgs, labels, target)
            We ignore 'target' (no supervision on I^t).
            labels (if provided) are NOT reconstruction targets.
        """

        if len(batch) == 2:
            imgs, _target = batch
            labels = None
        else:
            imgs, labels, _target = batch

        B, T = imgs.shape[:2]

        # -----------------------------
        # Encode / decode (CITRISVAE)
        # -----------------------------
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()

        x_rec = self.decoder(
            z_sample.unflatten(0, imgs.shape[:2])[:, 1:].flatten(0, 1)
        )

        z_sample, z_mean, z_logstd, x_rec = [
            t.unflatten(0, (imgs.shape[0], -1))
            for t in [z_sample, z_mean, z_logstd, x_rec]
        ]
        # shapes: (B, T, ...)

        # Reconstruction loss (MSE over pixels)
        if isinstance(self.decoder, nn.Identity):
            # x_rec은 (B, T-1, C, H, W) 형태를 기대하므로
            rec_loss = z_mean.new_zeros(B, max(T - 1, 1))
        else:
            rec_target = imgs[:, 1:]  # (B, T-1, C, H, W)

            # safety check
            assert x_rec.shape == rec_target.shape, \
                f"x_rec {x_rec.shape} vs rec_target {rec_target.shape}"

            rec_loss = F.mse_loss(
                x_rec, rec_target, reduction="none"
    ).sum(dim=[-3, -2, -1])  # (B, T-1)
        # -----------------------------
        # TS-CP^2 components
        # -----------------------------
        # CPC loss + change scores
        c, cpc_loss = self._cpc_and_change_scores(z_mean)  # c: (B, T-1)

        # soft gates g_t
        g = self._compute_gates(z_mean, c)  # (B, T-1, K)

        # gated transition prior KL
        kld_per_seq = self._gated_prior_kl(z_mean, z_logstd, g)  # (B,)

        # gate sparsity regularizer
        gate_loss = g.mean()  # 1/(T K) * sum_{t,i} g_t(i)

        # -----------------------------
        # Combine losses
        # -----------------------------
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        beta_t1 = getattr(self.hparams, "beta_t1", 1.0)

        base_loss = (
            rec_loss.sum(dim=1) + kld_factor * beta_t1 * kld_per_seq
        )  # (B,)
        base_loss = base_loss / max(T - 1, 1)

        loss = (
            base_loss.mean()
            + self.lambda_cpc * cpc_loss
            + self.lambda_gate * gate_loss
        )

        # Logging
        self.log(f"{mode}_kld_t1", kld_per_seq.mean() / max(T - 1, 1))
        self.log(f"{mode}_rec_loss_t1", rec_loss.mean())
        self.log(f"{mode}_cpc_loss", cpc_loss)
        self.log(f"{mode}_gate_loss", gate_loss)
        if mode == "train":
            self.log(f"{mode}_kld_scheduling", kld_factor)

        return loss

    # ------------------------------------------------------------------
    # Lightning hooks: reuse _get_loss for all splits
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode="val")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode="test")
        self.log("test_loss", loss)
