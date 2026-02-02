"""
EncryptedTTConv2d - Tensor Train decomposed encrypted 2D convolution layer.

Applies TT decomposition to the flattened convolution kernel for reduced
parameter count and memory-efficient encrypted inference.

References:
    Foundational TT Papers:
    - Oseledets, I. V. (2011). "Tensor-Train Decomposition".
      SIAM Journal on Scientific Computing, 33(5), 2295-2317.
      https://epubs.siam.org/doi/10.1137/090752286
      
    - Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. (2015).
      "Tensorizing Neural Networks". NeurIPS 2015.
      https://proceedings.neurips.cc/paper/5787-tensorizing-neural-networks
      
    CNN-Specific TT Papers:
    - Garipov, T., Podoprikhin, D., Novikov, A., & Vetrov, D. (2016).
      "Ultimate Tensorization: compressing convolutional and FC layers alike".
      https://arxiv.org/abs/1611.03214
      Key contribution: Extends TT to conv layers via kernel reshaping.
      
    - Gabor, M. & Zdunek, R. (2022). "Convolutional Neural Network Compression 
      via Tensor-Train Decomposition on Permuted Weight Tensor with Automatic 
      Rank Determination". ICCS 2022.
      https://link.springer.com/chapter/10.1007/978-3-031-08757-8_54
      Key contribution: Permutation-based TT for better CNN compression.
      
    - Wang, D., Zhao, G., Li, G., Deng, L., & Wu, Y. (2020).
      "Compressing 3DCNNs based on tensor train decomposition".
      Neural Networks, Volume 131.
      https://doi.org/10.1016/j.neunet.2020.07.028
      Key contribution: TT decomposition for higher-dimensional conv kernels.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import torch

from .module import EncryptedModule
from .tt_linear import _factorize, _pad_to_factorizable, _balance_factors

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _compute_energy_rank(S: torch.Tensor, threshold: float = 0.90) -> int:
    """Compute rank based on cumulative energy of singular values.
    
    E(r) = (Σ_{i=1}^{r} σ_i²) / (Σ_{all} σ_i²)
    
    Returns smallest r where E(r) >= threshold.
    
    Args:
        S: 1D tensor of singular values.
        threshold: Energy threshold (default: 0.90).
        
    Returns:
        Rank r where cumulative energy >= threshold.
    """
    if S.numel() == 0:
        return 1
    
    total_energy = torch.sum(S ** 2)
    if total_energy == 0:
        return 1
    
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    # Find first index where ratio >= threshold
    rank = int(torch.searchsorted(energy_ratio, torch.tensor(threshold, dtype=energy_ratio.dtype), right=True).item()) + 1
    rank = min(rank, S.numel())
    rank = max(rank, 1)
    
    return rank


def _tt_decompose_matrix(
    weight_flat: torch.Tensor,
    tt_shapes: List[Tuple[int, int]],
    max_rank: int,
    svd_threshold: float,
    use_energy: bool = False,
    energy_threshold: float = 0.90,
) -> List[torch.Tensor]:
    """Apply TT-SVD decomposition to a 2D weight matrix.
    
    Args:
        weight_flat: Weight matrix of shape (out_features, in_features).
        tt_shapes: List of (n_k, m_k) factor pairs.
        max_rank: Maximum TT-rank.
        svd_threshold: Threshold for SVD rank determination (used when use_energy=False).
        use_energy: If True, use energy-based rank determination instead of relative threshold.
        energy_threshold: Energy threshold for rank determination (used when use_energy=True).
        
    Returns:
        List of TT-cores.
    """
    num_cores = len(tt_shapes)
    
    # Reshape for TT decomposition
    out_factors = [n for n, _ in tt_shapes]
    in_factors = [m for _, m in tt_shapes]
    
    weight_tensor = weight_flat.reshape(*out_factors, *in_factors)
    permute_order = []
    for idx in range(num_cores):
        permute_order.append(idx)
        permute_order.append(num_cores + idx)
    weight_tensor = weight_tensor.permute(*permute_order).contiguous()
    weight_tensor = weight_tensor.reshape(*[n * m for n, m in tt_shapes])
    
    # TT-SVD decomposition
    tt_cores: List[torch.Tensor] = []
    r_prev = 1
    current = weight_tensor
    
    for k in range(num_cores - 1):
        n_k, m_k = tt_shapes[k]
        left_dim = r_prev * n_k * m_k
        right_dim = current.numel() // left_dim
        matrix = current.reshape(left_dim, right_dim)
        
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        if use_energy:
            rank = _compute_energy_rank(S, energy_threshold)
            rank = min(rank, max_rank, S.numel())
        else:
            if S.numel() == 0 or S[0] == 0:
                rank = 1
            else:
                rel = S / S[0]
                rank = int((rel >= svd_threshold).sum().item())
                if rank < 1:
                    rank = 1
            
            rank = min(rank, max_rank, S.numel())
        
        core = (U[:, :rank] * S[:rank]).reshape(r_prev, n_k * m_k, rank)
        tt_cores.append(core)
        
        current = Vh[:rank, :]
        r_prev = rank
    
    # Last core
    n_k, m_k = tt_shapes[-1]
    last_core = current.reshape(r_prev, n_k * m_k, 1)
    tt_cores.append(last_core)
    
    return tt_cores


def _reconstruct_from_cores(
    tt_cores: List[torch.Tensor],
    tt_shapes: List[Tuple[int, int]],
) -> torch.Tensor:
    """Reconstruct weight matrix from TT-cores.
    
    Args:
        tt_cores: List of TT-cores.
        tt_shapes: List of (n_k, m_k) factor pairs.
        
    Returns:
        Weight matrix of shape (out_features, in_features).
    """
    # Reshape each core to (r_prev, n_k, m_k, r_next)
    cores_4d = []
    for core, (n_k, m_k) in zip(tt_cores, tt_shapes):
        r_prev, _, r_next = core.shape
        cores_4d.append(core.reshape(r_prev, n_k, m_k, r_next))
    
    # Contract sequentially using einsum
    result = cores_4d[0][0]  # (n_1, m_1, r_1) - first core has r_0=1
    for core in cores_4d[1:]:
        result = torch.einsum("...r,rnmR->...nmR", result, core)
    
    # Remove trailing rank dimension (should be 1)
    result = result.squeeze(-1)
    
    # Permute from (n1, m1, n2, m2, ...) to (n1, n2, ..., m1, m2, ...)
    num_cores = len(tt_shapes)
    perm = list(range(0, 2 * num_cores, 2)) + list(range(1, 2 * num_cores, 2))
    result = result.permute(*perm).contiguous()
    
    # Reshape to (out_features, in_features)
    out_size = math.prod(n for n, _ in tt_shapes)
    in_size = math.prod(m for _, m in tt_shapes)
    return result.reshape(out_size, in_size)


def find_best_permutation(
    weight_4d: torch.Tensor,
    tt_shapes: List[Tuple[int, int]],
    max_rank: int,
    svd_threshold: float,
    padded_out: int,
    padded_in: int,
    energy_threshold: float = 0.90,
) -> Tuple[Tuple[int, ...], List[torch.Tensor], float]:
    """Find the best dimension permutation for TT decomposition of a 4D conv kernel.
    
    Tries all 6 permutations of the input dimensions (in_ch, kH, kW) while keeping
    out_ch as the first dimension, then selects the one with lowest reconstruction
    error. Uses energy-based rank determination for better compression.
    
    Args:
        weight_4d: 4D weight tensor of shape (out_ch, in_ch, kH, kW).
        tt_shapes: List of (n_k, m_k) factor pairs for TT decomposition.
        max_rank: Maximum TT-rank.
        svd_threshold: Threshold for SVD rank determination (fallback).
        padded_out: Padded output dimension for TT decomposition.
        padded_in: Padded input dimension for TT decomposition.
        energy_threshold: Energy threshold for rank determination.
        
    Returns:
        Tuple of:
        - best_permutation: Tuple of 4 ints representing the best dimension order.
        - best_cores: List of TT-cores from the best decomposition.
        - best_error: Relative Frobenius norm error of the best decomposition.
        
    References:
        Gabor, M. & Zdunek, R. (2022). "Convolutional Neural Network Compression 
        via Tensor-Train Decomposition on Permuted Weight Tensor with Automatic 
        Rank Determination". ICCS 2022.
    """
    out_ch, in_ch, kH, kW = weight_4d.shape
    original_in_size = in_ch * kH * kW
    original_flat = weight_4d.reshape(out_ch, original_in_size).to(torch.float64)
    original_norm = torch.norm(original_flat)
    
    if original_norm == 0:
        padded_weight = torch.zeros(padded_out, padded_in, dtype=torch.float64)
        cores = _tt_decompose_matrix(padded_weight, tt_shapes, max_rank, svd_threshold)
        return (0, 1, 2, 3), cores, 0.0
    
    # Generate all 6 permutations
    permutations = [(0,) + in_perm for in_perm in itertools.permutations([1, 2, 3])]
    
    def evaluate_permutation(perm_tuple):
        """Evaluate a single permutation."""
        try:
            permuted = weight_4d.permute(*perm_tuple).contiguous()
            flat = permuted.reshape(out_ch, original_in_size).to(torch.float64)
            
            padded_weight = torch.zeros(padded_out, padded_in, dtype=torch.float64)
            copy_out = min(out_ch, padded_out)
            copy_in = min(original_in_size, padded_in)
            padded_weight[:copy_out, :copy_in] = flat[:copy_out, :copy_in]
            
            cores = _tt_decompose_matrix(
                padded_weight, tt_shapes, max_rank, svd_threshold,
                use_energy=True, energy_threshold=energy_threshold
            )
            
            reconstructed = _reconstruct_from_cores(cores, tt_shapes)
            reconstructed_flat = reconstructed[:out_ch, :original_in_size]
            
            inv_perm = tuple(perm_tuple.index(i) for i in range(4))
            in_perm = perm_tuple[1:]
            reconstructed_4d = reconstructed_flat.reshape(out_ch, *[weight_4d.shape[i] for i in in_perm])
            unpermuted = reconstructed_4d.permute(*inv_perm).contiguous()
            unpermuted_flat = unpermuted.reshape(out_ch, original_in_size)
            
            error = torch.norm(original_flat - unpermuted_flat) / original_norm
            error_val = error.item()
            
            return (perm_tuple, cores, error_val)
        except Exception:
            return (perm_tuple, [], float('inf'))
    
    # Parallel evaluation with ThreadPoolExecutor
    # Using max 3 workers since we have 6 permutations and want to avoid GIL contention
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(evaluate_permutation, permutations))
    
    # Find best result
    best_perm: Tuple[int, ...] = (0, 1, 2, 3)
    best_cores: List[torch.Tensor] = []
    best_error = float('inf')
    
    for perm_tuple, cores, error_val in results:
        if error_val < best_error:
            best_error = error_val
            best_perm = perm_tuple
            best_cores = cores
    
    return best_perm, best_cores, best_error

class EncryptedTTConv2d(EncryptedModule):
    """Encrypted 2D convolution layer using Tensor Train decomposition.
    
    Decomposes the convolution kernel W from shape (out_ch, in_ch/groups, kH, kW) 
    into TT-cores by first flattening to (out_ch/groups, in_ch/groups * kH * kW), 
    then applying SVD-based TT decomposition per group. This reduces parameter 
    count and enables memory-efficient HE inference.
    
    The forward pass uses the same block-diagonal matrix multiplication
    approach as EncryptedConv2d for im2col-based convolution.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (possibly padded).
        kernel_size: Size of the convolution kernel (kH, kW).
        groups: Number of groups for grouped convolution. Default: 1
        dilation: Dilation of the convolution. Default: (1, 1)
        tt_cores_per_group: List of TT-cores for each group.
        tt_shapes: List of (n_k, m_k) factor pairs for each mode.
        bias: Optional bias vector.
        stride: Stride of the convolution. Default: (1, 1)
        padding: Padding added to input. Default: (0, 0)
        original_out_channels: Original (unpadded) output channels.
        original_in_size_per_group: Original in_channels/groups * kH * kW before padding.
        
    Example:
        >>> conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        >>> tt_conv = EncryptedTTConv2d.from_torch(conv)
        >>> enc_output = tt_conv(enc_input)
        
    References:
        - Novikov et al. (2015) "Tensorizing Neural Networks" - TT-decomposition for neural layers
        - Garipov et al. (2016) "Ultimate Tensorization" - TT for convolutional kernels
        - Oseledets (2011) "Tensor-Train Decomposition" - TT-SVD algorithm
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        groups: int = 1,
        dilation: Tuple[int, int] = (1, 1),
        tt_cores_per_group: Optional[List[List[torch.Tensor]]] = None,
        tt_shapes: Optional[List[Tuple[int, int]]] = None,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        original_out_channels: Optional[int] = None,
        original_in_size_per_group: Optional[int] = None,
        kernel_permutation: Optional[Tuple[int, ...]] = None,
        tt_cores: Optional[List[torch.Tensor]] = None,
        original_in_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._original_out_channels = original_out_channels or out_channels
        
        self.kernel_permutations_per_group: List[Optional[Tuple[int, ...]]] = []
        self._inverse_permutations_per_group: List[Optional[Tuple[int, ...]]] = []
        
        if kernel_permutation is not None:
            if type(kernel_permutation) is tuple:
                self.kernel_permutations_per_group = [kernel_permutation] * groups
            else:
                self.kernel_permutations_per_group = cast(List[Optional[Tuple[int, ...]]], list(kernel_permutation))
            
            self._inverse_permutations_per_group = [
                tuple(perm.index(i) for i in range(4)) if perm else None
                for perm in self.kernel_permutations_per_group
            ]
        else:
            self.kernel_permutations_per_group = [None] * groups
            self._inverse_permutations_per_group = [None] * groups
        
        self._kernel_permutation = self.kernel_permutations_per_group[0] if groups == 1 else None
        self._inverse_permutation = self._inverse_permutations_per_group[0] if groups == 1 else None
        
        # Handle legacy single-group case
        if tt_cores is not None and tt_cores_per_group is None:
            tt_cores_per_group = [tt_cores]
        if original_in_size is not None and original_in_size_per_group is None:
            original_in_size_per_group = original_in_size
            
        if tt_cores_per_group is None or tt_shapes is None:
            raise ValueError("tt_cores_per_group and tt_shapes are required")
            
        self.tt_shapes = tt_shapes
        in_ch_per_group = in_channels // groups
        self._original_in_size_per_group = original_in_size_per_group or (
            in_ch_per_group * kernel_size[0] * kernel_size[1]
        )
        
        # Store TT-cores per group as float64 for CKKS precision
        self.tt_cores_per_group: List[List[torch.Tensor]] = []
        for g, group_cores in enumerate(tt_cores_per_group):
            group_list: List[torch.Tensor] = []
            for i, core in enumerate(group_cores):
                core_param = core.detach().to(dtype=torch.float64, device="cpu")
                group_list.append(core_param)
                self.register_parameter(f"tt_core_g{g}_{i}", core_param)
            self.tt_cores_per_group.append(group_list)
        
        # Legacy attribute for backward compatibility
        self.tt_cores = self.tt_cores_per_group[0] if groups == 1 else self.tt_cores_per_group[0]
        
        # Pre-compute effective weight matrix
        self._effective_weight = self._reconstruct_weight()
        
        # Flattened kernel size for matmul (per group)
        self._flat_in_size_per_group = self._effective_weight.shape[1] // groups
        self._flat_in_size = self._effective_weight.shape[1]
        
        # Store bias
        if bias is not None:
            self.bias = bias.detach().to(dtype=torch.float64, device="cpu")
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
        
        # CNN layout tracking
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
    
    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix from TT-cores via tensor contraction.
        
        For grouped convolutions, returns a block-diagonal matrix where each
        block corresponds to one group's weight. Applies inverse permutation
        if permutation optimization was used.
        
        Returns:
            Weight matrix of shape (out_channels, in_channels * kH * kW) for groups=1,
            or block-diagonal for groups > 1.
        """
        out_per_group = self._original_out_channels // self.groups
        in_ch_per_group = self.in_channels // self.groups
        kH, kW = self.kernel_size
        
        if self.groups == 1:
            weight = _reconstruct_from_cores(self.tt_cores_per_group[0], self.tt_shapes)
            
            if self._inverse_permutation is not None and self._inverse_permutation != (0, 1, 2, 3):
                perm = self._kernel_permutation
                assert perm is not None
                permuted_shape = tuple([
                    out_per_group if perm[0] == 0 else (in_ch_per_group if perm[0] == 1 else (kH if perm[0] == 2 else kW)),
                    out_per_group if perm[1] == 0 else (in_ch_per_group if perm[1] == 1 else (kH if perm[1] == 2 else kW)),
                    out_per_group if perm[2] == 0 else (in_ch_per_group if perm[2] == 1 else (kH if perm[2] == 2 else kW)),
                    out_per_group if perm[3] == 0 else (in_ch_per_group if perm[3] == 1 else (kH if perm[3] == 2 else kW)),
                ])
                orig_in_size = in_ch_per_group * kH * kW
                weight_4d = weight[:out_per_group, :orig_in_size].reshape(*permuted_shape)
                unpermuted = weight_4d.permute(*self._inverse_permutation).contiguous()
                weight_2d = unpermuted.reshape(out_per_group, orig_in_size)
                
                padded_weight = torch.zeros_like(weight)
                padded_weight[:out_per_group, :orig_in_size] = weight_2d
                return padded_weight
            
            return weight
        
        padded_out_per_group = self.out_channels // self.groups
        padded_in_per_group = math.prod(m for _, m in self.tt_shapes)
        
        total_out = self.out_channels
        total_in = padded_in_per_group * self.groups
        
        full_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        
        for g in range(self.groups):
            group_weight = _reconstruct_from_cores(
                self.tt_cores_per_group[g], self.tt_shapes
            )
            row_start = g * padded_out_per_group
            row_end = (g + 1) * padded_out_per_group
            col_start = g * padded_in_per_group
            col_end = (g + 1) * padded_in_per_group
            full_weight[row_start:row_end, col_start:col_end] = group_weight
        
        return full_weight
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass on encrypted input using pre-computed effective weight.
        
        Args:
            x: Encrypted input tensor.
               - 2D with _cnn_layout: (num_patches, patch_features) - block-diagonal HE matmul
               - 1D: (flattened,) - single patch matmul
               
        Returns:
            Encrypted output after convolution.
            
        Raises:
            RuntimeError: If input is 4D, 3D, or 2D without CNN layout.
        """
        input_ndim = len(x.shape)
        
        if input_ndim == 4:
            raise RuntimeError(
                "EncryptedTTConv2d does not support 4D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout."
            )
        
        elif input_ndim == 3:
            raise RuntimeError(
                "EncryptedTTConv2d does not support 3D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout."
            )
        
        elif input_ndim == 2:
            # 2D input: (num_patches, patch_features) - pre-unfolded via encrypt_cnn_input
            if hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
                return self._forward_he_packed(x)
            else:
                raise RuntimeError(
                    "EncryptedTTConv2d requires CNN layout for 2D input. "
                    "Pre-process input using ctx.encrypt_cnn_input() to set "
                    "_cnn_layout metadata."
                )
        
        elif input_ndim == 1:
            # 1D input: single flattened patch
            self.input_shape = x.shape
            self.output_shape = (self._original_out_channels,)
            return x.matmul(self._effective_weight, self.bias)
        
        else:
            raise ValueError(
                f"Expected 1D-4D input, got {input_ndim}D with shape {x.shape}"
            )
    
    def _forward_he_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass using block-diagonal HE matmul for packed im2col patches.
        
        Input: flattened patches of shape (num_patches * patch_features,)
        Output: flattened output of shape (num_patches * out_channels,)
        
        Args:
            x: Encrypted im2col patches with _cnn_layout metadata.
            
        Returns:
            Encrypted output of shape (num_patches * out_channels,).
        """
        layout = x._cnn_layout
        num_patches = layout['num_patches']
        patch_features = layout['patch_features']
        
        self.input_shape = (num_patches, patch_features)
        self.output_shape = (num_patches, self._original_out_channels)
        
        total_out = num_patches * self._original_out_channels
        total_in = num_patches * patch_features
        
        # Update shape for matmul to understand the flat structure
        x._shape = (total_in,)
        
        # Create block-diagonal weight matrix using original (unpadded) dimensions
        original_in_size = self._original_in_size_per_group * self.groups
        weight_matrix = self._effective_weight[:self._original_out_channels, :original_in_size]
        
        block_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        for p in range(num_patches):
            row_start = p * self._original_out_channels
            row_end = (p + 1) * self._original_out_channels
            col_start = p * patch_features
            col_end = (p + 1) * patch_features
            block_weight[row_start:row_end, col_start:col_end] = weight_matrix
        
        # Create block-diagonal bias
        if self.bias is not None:
            block_bias = self.bias[:self._original_out_channels].repeat(num_patches)
        else:
            block_bias = None
        
        # Use standard matmul with block-diagonal weights
        result = x.matmul(block_weight, block_bias)
        
        # Update CNN layout for next layer
        result._cnn_layout = {
            'num_patches': num_patches,
            'patch_features': self._original_out_channels,
            'original_shape': layout.get('original_shape'),
        }
        result._shape = (num_patches, self._original_out_channels)
        
        return result
    
    def mult_depth(self) -> int:
        """Convolution uses 1 multiplication (single matmul with effective weight)."""
        return 1
    
    def get_output_size(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int]:
        """Compute the output spatial dimensions.
        
        Args:
            input_height: Height of the input.
            input_width: Width of the input.
            
        Returns:
            Tuple of (output_height, output_width).
        """
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        
        # Compute effective kernel size with dilation
        effective_kH = dH * (kH - 1) + 1
        effective_kW = dW * (kW - 1) + 1
        
        out_h = (input_height + 2 * pH - effective_kH) // sH + 1
        out_w = (input_width + 2 * pW - effective_kW) // sW + 1
        
        return out_h, out_w
    
    @classmethod
    def from_torch(
        cls,
        conv: torch.nn.Conv2d,
        max_rank: Optional[int] = None,
        svd_threshold: float = 1e-6,
        TT: bool = False,
        energy_threshold: float = 0.90,
    ) -> Optional["EncryptedTTConv2d"]:
        """Create from a PyTorch Conv2d layer using TT decomposition.
        
        The 4D kernel (out_ch, in_ch/groups, kH, kW) is flattened to 2D 
        (out_ch/groups, in_ch/groups * kH * kW), then decomposed using SVD-based TT.
        For grouped convolutions, each group is decomposed independently.
        
        Args:
            conv: The PyTorch Conv2d layer to convert.
            max_rank: Maximum TT-rank (auto-determined if None).
            svd_threshold: Threshold for SVD rank determination (default: 1e-6).
            TT: If True, try all 6 input dimension orderings and select the one
                with lowest reconstruction error. Based on Gabor & Zdunek (2022).
                Default: False.
            energy_threshold: Energy threshold for rank determination when TT=True.
                E(r) = (sum_{i=1}^{r} sigma_i^2) / (sum_{all} sigma_i^2)
                Default: 0.90.
            
        Returns:
            EncryptedTTConv2d, or None if layer is too small for TT.
            
        References:
            - Novikov et al. (2015) "Tensorizing Neural Networks"
            - Garipov et al. (2016) "Ultimate Tensorization"
            - Gabor & Zdunek (2022) "CNN Compression via TT on Permuted Weight Tensor"
        """
        # Extract parameters
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        groups = conv.groups
        kernel_size = cast(Tuple[int, int], conv.kernel_size)
        stride = cast(Tuple[int, int], conv.stride)
        
        dilation = conv.dilation
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        dilation = cast(Tuple[int, int], dilation)
        
        # Handle 'same' padding
        if isinstance(conv.padding, str):
            if conv.padding == 'same':
                padding: Tuple[int, int] = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = (0, 0)
        else:
            padding = cast(Tuple[int, int], conv.padding)
        
        # Validate grouped conv dimensions
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )
        
        # Per-group dimensions
        in_per_group = in_channels // groups
        out_per_group = out_channels // groups
        
        # Flatten kernel per group: (out_ch/g, in_ch/g * kH * kW)
        flat_in_per_group = in_per_group * kernel_size[0] * kernel_size[1]
        params_per_group = out_per_group * flat_in_per_group
        
        # Threshold check (per group)
        if params_per_group < 1024:
            return None
        
        # Pad dimensions for factorization (per group)
        padded_out_per_group = _pad_to_factorizable(out_per_group)
        padded_in_per_group = _pad_to_factorizable(flat_in_per_group)
        
        out_factors = _factorize(padded_out_per_group)
        in_factors = _factorize(padded_in_per_group)
        out_factors, in_factors = _balance_factors(out_factors, in_factors)
        
        if not out_factors or not in_factors:
            return None
        
        padded_out_per_group = math.prod(out_factors)
        padded_in_per_group = math.prod(in_factors)
        
        tt_shapes = list(zip(out_factors, in_factors))
        
        # TT-SVD parameters
        rank_cap = 64 if max_rank is None else min(64, max_rank)
        rank_cap = max(1, rank_cap)
        
        weight = conv.weight.detach().to(dtype=torch.float64, device="cpu")
        
        tt_cores_per_group: List[List[torch.Tensor]] = []
        kernel_permutation: Optional[Tuple[int, ...]] = None
        
        for g in range(groups):
            group_weight_4d = weight[g * out_per_group:(g + 1) * out_per_group]
            
            if TT:
                best_perm, group_cores, _ = find_best_permutation(
                    group_weight_4d,
                    tt_shapes,
                    rank_cap,
                    svd_threshold,
                    padded_out_per_group,
                    padded_in_per_group,
                    energy_threshold,
                )
                if g == 0:
                    kernel_permutation = best_perm
            else:
                group_weight_flat = group_weight_4d.reshape(out_per_group, flat_in_per_group)
                
                if padded_in_per_group != flat_in_per_group or padded_out_per_group != out_per_group:
                    padded_weight = torch.zeros(
                        padded_out_per_group, padded_in_per_group, dtype=torch.float64
                    )
                    padded_weight[:out_per_group, :flat_in_per_group] = group_weight_flat
                    group_weight_flat = padded_weight
                
                group_cores = _tt_decompose_matrix(
                    group_weight_flat, tt_shapes, rank_cap, svd_threshold
                )
            
            tt_cores_per_group.append(group_cores)
        
        # Handle bias
        bias_param = getattr(conv, "bias", None)
        bias = (
            bias_param.detach().to(dtype=torch.float64, device="cpu")
            if bias_param is not None
            else None
        )
        
        # Pad bias if needed
        total_padded_out = padded_out_per_group * groups
        if bias is not None and total_padded_out != out_channels:
            padded_bias = torch.zeros(total_padded_out, dtype=torch.float64)
            # Copy bias for each group
            for g in range(groups):
                src_start = g * out_per_group
                src_end = (g + 1) * out_per_group
                dst_start = g * padded_out_per_group
                dst_end = dst_start + out_per_group
                padded_bias[dst_start:dst_end] = bias[src_start:src_end]
            bias = padded_bias
        elif bias is None and total_padded_out != out_channels:
            bias = torch.zeros(total_padded_out, dtype=torch.float64)
        
        return cls(
            in_channels=in_channels,
            out_channels=total_padded_out,
            kernel_size=kernel_size,
            groups=groups,
            dilation=dilation,
            tt_cores_per_group=tt_cores_per_group,
            tt_shapes=tt_shapes,
            bias=bias,
            stride=stride,
            padding=padding,
            original_out_channels=out_channels,
            original_in_size_per_group=flat_in_per_group,
            kernel_permutation=kernel_permutation,
        )
    
    def extra_repr(self) -> str:
        ranks = [self.tt_cores_per_group[0][0].shape[0]] + [
            c.shape[2] for c in self.tt_cores_per_group[0]
        ]
        s = (
            f"{self.in_channels}, {self._original_out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, "
            f"num_cores={len(self.tt_cores_per_group[0])}, ranks={ranks}, "
            f"bias={self.bias is not None}"
        )
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        return s
