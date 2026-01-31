"""
Sparse Autoencoder (SAE) model for training on GraphCast layer activations.

This module implements a Top-K Sparse Autoencoder with AuxK loss following the
OpenAI recipe. It includes:
- A PyTorch IterableDataset for loading activation data from .npy files
- The SAE model with dead neuron tracking and gradient projection
- Utilities for converting PyTorch models to JAX format

Source: https://github.com/theodoremacmillan/graphcast-interpretability/blob/main/src/graphcast_interpretability/model.py
"""
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from typing import Tuple, Optional, Union, List
from walrus_workshop.utils import filter_kwargs

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_float32_matmul_precision("high")

print(f"Using device: {device}")


class NPYLayerActivationStream(IterableDataset):
    """
    Iterable PyTorch dataset for per-layer GraphCast activations saved as .npy arrays.

    This dataset loads activation data from .npy files (one timestep per file) and
    yields batches of node activations. It automatically handles both [N, D] and
    [N, 1, D] input shapes by removing singleton dimensions. The dataset supports
    multi-worker data loading with proper sharding and shuffling.

    Attributes:
        files: List of paths to .npy files matching the layer prefix.
        d_in: Input dimension (number of features per node).
        batch: Batch size for yielding data.
        seed: Random seed for reproducibility.
        steps_per_epoch: Number of batches to yield per epoch (None = all batches).
        file_meta: List of metadata dictionaries containing file paths and node counts.
        total_batches: Total number of batches available across all files.

    Example:
        >>> dataset = NPYLayerActivationStream(
        ...     data_dirs=["/path/to/activations"],
        ...     layer_prefix="layer0012_",
        ...     d_in=512,
        ...     batch_size=8192
        ... )
        >>> loader = DataLoader(dataset, num_workers=4)
        >>> for batch in loader:
        ...     # batch shape: [batch_size, d_in]
        ...     pass
    """

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        layer_prefix: str = "layer0012_",
        d_in: int = 512,
        batch_size: int = 8192,
        steps_per_epoch: Optional[int] = None,
        seed: int = 0
    ):
        """
        Initialize the activation stream dataset.

        Args:
            data_dirs: Directory or list of directories containing .npy activation files.
            layer_prefix: Prefix pattern to match layer files (e.g., "layer0012_").
                Files matching "{layer_prefix}*.npy" will be loaded.
            d_in: Expected input dimension (number of features per node).
            batch_size: Number of nodes to include in each batch.
            steps_per_epoch: Number of batches to yield per epoch. If None, yields
                all available batches based on total nodes and batch_size.
            seed: Random seed for shuffling and reproducibility.

        Raises:
            FileNotFoundError: If no .npy files are found matching the layer prefix.
            AssertionError: If any file has an unexpected shape.
        """
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # --- Collect matching files ---
        file_list = []
        for root in data_dirs:
            pattern = os.path.join(root, f"{layer_prefix}*.npy")
            file_list.extend(glob.glob(pattern))
        file_list = sorted(file_list)
        if not file_list:
            raise FileNotFoundError(f"No .npy files found matching {layer_prefix} in {data_dirs}")

        self.files = file_list
        self.d_in = d_in
        self.batch = batch_size
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch

        # --- Compute metadata ---
        self.file_meta = []
        for f in self.files:
            try:
                arr = np.load(f, mmap_mode="r")
            except ValueError as e:
                print(f"[CORRUPT] {f}: {e}")
                continue
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]           # ✅ fix singleton dimension
            assert arr.ndim == 2 and arr.shape[1] == d_in, f"{f} has wrong shape {arr.shape}"
            self.file_meta.append({"path": f, "n_nodes": int(arr.shape[0])})

        total_nodes = sum(m["n_nodes"] for m in self.file_meta)
        self.total_batches = math.ceil(total_nodes / self.batch)
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.total_batches

    def __iter__(self):
        """
        Iterate over batches of activation data.

        This method handles multi-worker data loading by sharding files across workers
        and shuffling within each worker's shard. Each file is loaded, nodes are
        randomly permuted, and then batched.

        Yields:
            torch.Tensor: Batch of activations with shape [batch_size, d_in].
                The actual batch size may be smaller than self.batch for the last
                batch in a file or epoch.
        """
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        file_shard = self.file_meta[wid::nw]
        rng.shuffle(file_shard)

        max_batches_worker = math.ceil(self.steps_per_epoch / nw)
        batches_yielded = 0

        for md in file_shard:
            if batches_yielded >= max_batches_worker:
                break

            fpath = md["path"]
            X = np.load(fpath, mmap_mode="r")
            if X.ndim == 3 and X.shape[1] == 1:
                X = X[:, 0, :]                # ✅ fix again at runtime

            n = X.shape[0]
            perm = rng.permutation(n)
            for start in range(0, n, self.batch):
                if batches_yielded >= max_batches_worker:
                    return
                sel = perm[start:start + self.batch]
                if sel.size == 0:
                    break
                xb = torch.from_numpy(X[sel, :])
                yield xb
                batches_yielded += 1


def topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep top-k values per row, zero out the rest.

    This function performs a top-k sparsification operation on a tensor, keeping
    only the k largest values in each row and setting all other values to zero.
    This is used in the SAE to enforce sparsity in the latent code.

    Args:
        x: Input tensor of shape [batch_size, num_features].
        k: Number of top values to keep per row. If k >= num_features, returns
            the original tensor unchanged.

    Returns:
        torch.Tensor: Tensor of the same shape as x with only top-k values per
            row retained, all others set to zero.

    Example:
        >>> x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]])
        >>> topk(x, k=2)
        tensor([[0., 5., 3., 0.],
                [0., 0., 6., 3.]])
    """
    if k >= x.shape[1]:
        return x
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return x * mask


class SAE(nn.Module):
    """
    Top-K Sparse Autoencoder with AuxK loss following the OpenAI recipe.

    This SAE implements a sparse autoencoder that:
    - Encodes inputs into a sparse latent representation (top-k activations)
    - Decodes back to the input space
    - Tracks and handles dead neurons (neurons that never activate)
    - Uses an auxiliary reconstruction loss for dead neurons to encourage their use

    The model normalizes inputs per-sample (zero-mean, unit-norm) and uses a shared
    pre/post bias term. The decoder weights can be optionally kept at unit norm.

    Attributes:
        enc: Encoder linear layer (no bias).
        dec: Decoder linear layer (no bias).
        b_pre: Shared pre/post bias parameter.
        k: Number of active (top-k) features to keep in the latent code.
        k_aux: Number of top-k features to use in auxiliary reconstruction for dead neurons.
        unit_norm_decoder: Whether to normalize decoder columns to unit norm.
        dead_window: Number of consecutive inactive batches before a neuron is marked dead.
        miss_counts: Buffer tracking consecutive inactive batches per latent dimension.
        dead_mask: Boolean mask indicating which latent dimensions are dead.

    Example:
        >>> model = SAE(d_in=512, latent=8192, k_active=32, k_aux=8)
        >>> x = torch.randn(128, 512)
        >>> recon, code, aux_recon = model(x)
        >>> # recon: [128, 512] - main reconstruction
        >>> # code: [128, 8192] - sparse latent code (top-k only)
        >>> # aux_recon: [128, 512] - auxiliary reconstruction from dead neurons
    """
    def __init__(
        self,
        d_in: int,
        latent: int,
        k_active: int,
        k_aux: int,
        unit_norm_decoder: bool = True,
        dead_window: int = 3_000_000
    ):
        """
        Initialize the Sparse Autoencoder.

        Args:
            d_in: Input dimension (number of features per sample).
            latent: Latent dimension (number of dictionary features).
            k_active: Number of top-k features to keep active in the main code.
            k_aux: Number of top-k features to use in auxiliary reconstruction
                for dead neurons (helps revive them).
            unit_norm_decoder: If True, decoder columns are normalized to unit norm
                during forward pass. This is the recommended setting.
            dead_window: Number of consecutive batches a neuron must be inactive
                before being marked as dead. Dead neurons are excluded from main
                reconstruction but can participate in auxiliary reconstruction.
        """
        super().__init__()
        # --- Core layers (no internal bias terms) ---
        self.enc = nn.Linear(d_in, latent, bias=False)
        self.dec = nn.Linear(latent, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))  # shared pre/post bias

        # --- Hyperparameters ---
        self.d_in = d_in
        self.latent = latent
        self.k = k_active
        self.k_aux = k_aux
        self.unit_norm_decoder = unit_norm_decoder
        self.dead_window = dead_window
        self.eps = 1e-8

        # --- Dead neuron tracking ---
        self.register_buffer("miss_counts", torch.zeros(latent, dtype=torch.long))
        self.register_buffer("dead_mask", torch.zeros(latent, dtype=torch.bool))
        # self.register_buffer("miss_counts", (self.dead_window+1) * torch.ones(latent, dtype=torch.long))
        # self.register_buffer("dead_mask", torch.zeros(latent, dtype=torch.bool))

        # --- OpenAI-style initialization ---
        with torch.no_grad():
            # 1. Randomly initialize decoder (dictionary) and normalize columns
            torch.nn.init.normal_(self.dec.weight, mean=0.0, std=1.0)
            W = self.dec.weight
            W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))

            # 2. Set encoder = decoderᵀ
            self.enc.weight.copy_(W.t())

            # 3. Zero shared bias
            self.b_pre.zero_()

    def get_config(self):
        return dict(
            d_in=self.d_in,
            latent=self.latent,
            k_active=self.k,
            k_aux=self.k_aux,
            dead_window=self.dead_window,
        )

    def _renorm_decoder_columns_(self):
        """
        Renormalize decoder columns to unit L2 norm (in-place).

        This method ensures each column of the decoder weight matrix has unit norm,
        which is important for maintaining the dictionary structure. This is typically
        called after gradient updates if unit_norm_decoder is enabled.
        """
        W = self.dec.weight.data
        norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
        W.div_(norms)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the sparse autoencoder.

        The forward pass:
        1. Normalizes inputs per-sample (zero-mean, unit-norm)
        2. Encodes to sparse latent code (top-k activations)
        3. Decodes to reconstruction
        4. Computes auxiliary reconstruction from dead neurons

        Args:
            x: Input tensor of shape [batch_size, d_in].

        Returns:
            Tuple containing:
                - recon: Main reconstruction tensor of shape [batch_size, d_in].
                - code: Sparse latent code tensor of shape [batch_size, latent]
                    with only top-k values per sample non-zero.
                - aux_recon: Auxiliary reconstruction tensor of shape [batch_size, d_in]
                    computed from dead neurons only. Used for auxiliary loss to
                    encourage dead neuron revival.
        """
        # ---- Normalize inputs (zero-mean, unit-norm per sample) ----
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

        # ---- Subtract shared pre-bias before encoding ----
        x_bar = x - self.b_pre

        # ---- Encode ----
        code_pre = torch.relu(self.enc(x_bar))
        code = topk(code_pre, self.k)

        # ---- Decode and add shared bias back ----
        if self.unit_norm_decoder:
            W = self.dec.weight
            norms = W.norm(dim=0, keepdim=True).clamp_min(self.eps)
            recon = torch.addmm(self.b_pre, code, (W / norms).t())
        else:
            recon = self.dec(code) + self.b_pre

        # ---- AuxK reconstruction (dead latents only) ----
        dead_mask = self.dead_mask
        if dead_mask.any():
            dead_code = code_pre * dead_mask.unsqueeze(0)
            aux_code = topk(dead_code, min(self.k_aux, dead_code.shape[1]))
            aux_recon = torch.addmm(
                torch.zeros_like(self.b_pre),
                aux_code,
                self.dec.weight.t(),
            )
        else:
            aux_recon = torch.zeros_like(x)

        return recon, code, aux_recon

    @torch.no_grad()
    def update_dead_mask(self, code: torch.Tensor, batch_size: int) -> None:
        """
        Update dead neuron tracking.

        This method should be called after each forward pass during training to
        track which latent dimensions are inactive. Neurons that remain inactive
        for `dead_window` consecutive batches are marked as dead and excluded from
        the main reconstruction (but can still participate in auxiliary reconstruction).

        Args:
            code: Sparse latent code from forward pass, shape [batch_size, latent].
            batch_size: Number of samples in the current batch (used for counting
                inactive batches).
        """
        active = (code > 0).any(dim=0).cpu()
        self.miss_counts[active] = 0
        self.miss_counts[~active] += batch_size
        self.dead_mask = self.miss_counts >= self.dead_window


@torch.no_grad()
def _project_decoder_grads_orthogonal(model: SAE) -> None:
    """
    Project decoder gradients to be orthogonal to decoder columns (in-place).

    This function modifies the decoder gradients so that gradient updates don't
    change the column norms. This is useful when maintaining unit-norm decoder
    columns, as it ensures the gradient step preserves the normalization constraint.

    The projection removes the component of the gradient parallel to each column,
    leaving only the orthogonal component.

    Args:
        model: SAE model whose decoder gradients will be modified.
    """
    W = model.dec.weight
    G = model.dec.weight.grad
    if G is None:
        return
    dots = (G * W).sum(dim=0, keepdim=True)
    norms2 = (W * W).sum(dim=0, keepdim=True).clamp_min(1e-8)
    G.sub_((dots / norms2) * W)


@torch.no_grad()
def _renorm_decoder_columns_(model: SAE) -> None:
    """
    Renormalize decoder columns to unit L2 norm (in-place).

    This function ensures each column of the decoder weight matrix has unit norm.
    It's typically called after gradient updates when maintaining unit-norm decoder
    columns, either as an alternative to or in combination with gradient projection.

    Args:
        model: SAE model whose decoder weights will be renormalized.
    """
    W = model.dec.weight.data
    norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W.div_(norms)

# ---------- Convert PyTorch SAE to JAX format ----------


def load_sae_params_from_torch(
    ckpt_path: str,
    unit_norm_decoder: bool,
    k_active: int
):
    """
    Load PyTorch SAE checkpoint and convert to JAX-compatible format.

    This function loads a PyTorch SAE model checkpoint and converts it to a
    JAX-compatible dataclass format. The weights are transposed to match JAX
    conventions (JAX uses [out_features, in_features] while PyTorch uses
    [in_features, out_features] for Linear layers).

    Args:
        ckpt_path: Path to the PyTorch checkpoint file (.pt or .pth).
        unit_norm_decoder: Whether the decoder columns were normalized to unit
            norm in the original model.
        k_active: Number of top-k active features used in the original model.

    Returns:
        SAEStaticParams: Dataclass containing:
            - enc_w: Encoder weights as JAX array, shape [d_in, latent].
            - dec_w: Decoder weights as JAX array, shape [latent, d_in].
            - b_pre: Shared pre/post bias as JAX array, shape [d_in].
            - k_active: Number of top-k active features.
            - unit_norm_decoder: Whether decoder columns are unit-normalized.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        KeyError: If required keys are missing from the checkpoint state dict.

    Example:
        >>> params = load_sae_params_from_torch(
        ...     ckpt_path="model.pt",
        ...     unit_norm_decoder=True,
        ...     k_active=32
        ... )
        >>> # Use params.enc_w, params.dec_w, etc. in JAX code
    """
    import torch
    import jax.numpy as jnp
    from dataclasses import dataclass

    @dataclass
    class SAEStaticParams:
        """
        Static parameters for SAE model in JAX format.

        Attributes:
            enc_w: Encoder weight matrix, shape [d_in, latent].
            dec_w: Decoder weight matrix, shape [latent, d_in].
            b_pre: Shared pre/post bias vector, shape [d_in].
            k_active: Number of top-k active features to keep.
            unit_norm_decoder: Whether decoder columns are unit-normalized.
        """
        enc_w: jnp.ndarray
        dec_w: jnp.ndarray
        b_pre: jnp.ndarray
        k_active: int
        unit_norm_decoder: bool

    # --- load and unwrap nested dict ---
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]

    # --- extract weights ---
    enc_w_torch = state["enc.weight"]    # [latent, d_in]
    dec_w_torch = state["dec.weight"]    # [d_in, latent]
    b_pre_torch = state["b_pre"]         # [d_in]

    # --- transpose and convert to numpy/JAX ---
    enc_w = enc_w_torch.t().contiguous().cpu().numpy()  # [d_in, latent]
    dec_w = dec_w_torch.t().contiguous().cpu().numpy()  # [latent, d_in]
    b_pre = b_pre_torch.contiguous().cpu().numpy()      # [d_in]

    return SAEStaticParams(
        enc_w=jnp.asarray(enc_w),
        dec_w=jnp.asarray(dec_w),
        b_pre=jnp.asarray(b_pre),
        k_active=k_active,
        unit_norm_decoder=unit_norm_decoder,
    )


def load_sae(save_path, load_weights=True):
    # 1. Load
    checkpoint = torch.load(save_path)
    config = checkpoint["config"]

    config = filter_kwargs(config, SAE)

    # 2. Instantiate (The Pythonic Way)
    model = SAE(**config)

    if load_weights:
        # 3. Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

    return model, config