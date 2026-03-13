"""
EncryptedApproxAttention - Approximate attention using polynomial softmax.

For CKKS encrypted inference, exact softmax is not possible.
We approximate softmax using Taylor expansion of exp(x).
All attention computations run in pure HE using cipher-cipher operations.
"""

from __future__ import annotations

from collections import OrderedDict
import functools
import math
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import numpy as np
import torch
from numpy.polynomial.chebyshev import cheb2poly, chebfit
import cukks.batching as batching_module

from .module import EncryptedModule
from cukks.stats.crypto_reciprocal import _compute_reciprocal_coeffs

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _taylor_exp_coeffs(degree: int) -> List[float]:
    """Compute Taylor series coefficients for exp(x).
    
    exp(x) ≈ 1 + x + x²/2! + x³/3! + ... + x^n/n!
    
    Args:
        degree: Maximum degree of polynomial.
        
    Returns:
        List of coefficients [1, 1, 1/2!, 1/3!, ..., 1/n!].
    """
    coeffs = []
    factorial = 1.0
    for i in range(degree + 1):
        if i > 0:
            factorial *= i
        coeffs.append(1.0 / factorial)
    return coeffs


@functools.lru_cache(maxsize=8)
def _gaussian_exp_coeffs(domain: tuple[float, float], degree: int) -> List[float]:
    a, b = domain

    def g(t: float) -> float:
        x = (b - a) * (t + 1) / 2 + a
        return float(math.exp(-x))

    n_nodes = 100
    nodes = np.cos((2 * np.arange(n_nodes) + 1) * np.pi / (2 * n_nodes))
    values = np.array([g(t) for t in nodes])

    cheb_coeffs = chebfit(nodes, values, degree)
    power_coeffs = cheb2poly(cheb_coeffs)
    return power_coeffs.tolist()


class EncryptedApproxAttention(EncryptedModule):
    """Approximate multi-head attention for encrypted inference.
    
    Uses polynomial approximation for softmax since CKKS only supports
    polynomial operations. The softmax is approximated using Taylor
    expansion of exp(x).
    
    Supports both single-token (seq_len=1) and multi-token (seq_len>1) attention:
    - seq_len=1: Uses Taylor polynomial softmax approximation
    - seq_len>1: Uses Power-Softmax (p=2) with crypto_reciprocal for normalization
      Maximum seq_len is 8 due to multiplicative depth constraints.
    
    Note:
        This is an approximation - accuracy depends on input range
        and polynomial degree. Best results when attention scores
        are normalized to a small range (e.g., [-2, 2]).
    
    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        softmax_degree: Degree of polynomial for exp approximation.
        
    Example:
        >>> attn = EncryptedApproxAttention(embed_dim=64, num_heads=4)
        >>> output = attn(query, key, value)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        softmax_degree: int = 4,
        normalization_mode: str = "power_softmax",
        gaussian_gamma: float = 0.25,
    ) -> None:
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax_degree = softmax_degree
        if normalization_mode not in {"power_softmax", "gaussian"}:
            raise ValueError(
                "normalization_mode must be 'power_softmax' or 'gaussian', "
                f"got {normalization_mode!r}"
            )
        if gaussian_gamma <= 0.0:
            raise ValueError(f"gaussian_gamma must be positive, got {gaussian_gamma}")
        self.normalization_mode = normalization_mode
        self.gaussian_gamma = gaussian_gamma
        self._gaussian_domain = (0.0, 4.0)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Taylor coefficients for exp(x) approximation
        self._exp_coeffs = _taylor_exp_coeffs(softmax_degree)
        self._gaussian_exp_coeffs = _gaussian_exp_coeffs(self._gaussian_domain, softmax_degree)
        self._reciprocal_degree = 15
        self._packed_metadata_cache: OrderedDict[
            tuple[int, int, int, int, int, int],
            tuple[list[list[float]], list[list[float]], list[list[float]], tuple[int, ...], torch.Tensor],
        ] = OrderedDict()
        
        # Projection weights (initialized as identity for now)
        # These can be set via from_torch or manually
        self.q_weight: Optional[torch.Tensor] = None
        self.k_weight: Optional[torch.Tensor] = None
        self.v_weight: Optional[torch.Tensor] = None
        self.out_weight: Optional[torch.Tensor] = None
        
        self.q_bias: Optional[torch.Tensor] = None
        self.k_bias: Optional[torch.Tensor] = None
        self.v_bias: Optional[torch.Tensor] = None
        self.out_bias: Optional[torch.Tensor] = None
    
    def _apply_projection(
        self,
        x: "EncryptedTensor",
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> "EncryptedTensor":
        """Apply linear projection if weights are set."""
        if weight is None:
            return x
        return x.matmul(weight, bias)

    def _packed_token_mask(self, x: "EncryptedTensor", token_index: int, seq_len: int) -> List[float]:
        batch_size = getattr(x, "_batch_size", None)
        slots_per_sample = getattr(x, "_slots_per_sample", None)
        if batch_size is None or slots_per_sample is None:
            raise RuntimeError("Packed attention requires packed batch metadata")

        embed_dim = self.embed_dim
        if slots_per_sample != seq_len * embed_dim:
            raise RuntimeError(
                f"Packed attention expected slots_per_sample={seq_len * embed_dim}, got {slots_per_sample}."
            )

        slot_count = x._cipher.size
        active_size = batch_size * slots_per_sample
        start_in_sample = token_index * embed_dim
        mask = [0.0] * slot_count
        for batch_idx in range(batch_size):
            block_start = batch_idx * slots_per_sample + start_in_sample
            for offset in range(embed_dim):
                pos = block_start + offset
                if pos < active_size:
                    mask[pos] = 1.0
        return mask

    def _precompute_packed_metadata(
        self,
        x: "EncryptedTensor",
        seq_len: int,
        embed_dim: int,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]], tuple[int, ...], torch.Tensor]:
        batch_size = getattr(x, "_batch_size", None)
        slots_per_sample = getattr(x, "_slots_per_sample", None)
        if batch_size is None or slots_per_sample is None:
            raise RuntimeError("Packed attention requires packed batch metadata")

        cache_key = (x._cipher.size, batch_size, slots_per_sample, seq_len, embed_dim, self.num_heads)
        cached = self._packed_metadata_cache.get(cache_key)
        if cached is not None:
            self._packed_metadata_cache.move_to_end(cache_key)
            return cached

        query_masks = [self._packed_token_mask(x, token_index, seq_len) for token_index in range(seq_len)]
        power_outside_fills = [
            [-2.0 * (1.0 - value) for value in query_mask]
            for query_mask in query_masks
        ]
        gaussian_distance_cap = self._gaussian_domain[1] / self.gaussian_gamma
        gaussian_outside_fills = [
            [gaussian_distance_cap * (1.0 - value) for value in query_mask]
            for query_mask in query_masks
        ]
        rotation_offsets = tuple(sorted({
            (key_index - query_index) * embed_dim
            for query_index in range(seq_len)
            for key_index in range(seq_len)
        }))
        ones_block = torch.ones(embed_dim, embed_dim, dtype=torch.float64)
        cached = (query_masks, power_outside_fills, gaussian_outside_fills, rotation_offsets, ones_block)
        if len(self._packed_metadata_cache) >= 8:
            self._packed_metadata_cache.popitem(last=False)
        self._packed_metadata_cache[cache_key] = cached
        return cached

    def _validate_stip_layout(self, x: "EncryptedTensor") -> tuple[int, int]:
        layout = getattr(x, "_packing_layout", None)
        if layout is None:
            raise RuntimeError("STIP attention requires packing layout metadata")
        if not getattr(x, "_stip_layout_fresh", False):
            raise RuntimeError("STIP attention requires fresh sequence-layout provenance")
        if layout.d_model != self.embed_dim:
            raise RuntimeError(
                f"STIP attention expected d_model={self.embed_dim}, got {layout.d_model}."
            )
        if layout.num_heads != self.num_heads:
            raise RuntimeError(
                f"STIP attention expected num_heads={self.num_heads}, got {layout.num_heads}."
            )
        if x.shape != (layout.seq_len, layout.d_model):
            raise RuntimeError(
                "STIP attention requires tensor shape matching layout. "
                f"Got tensor shape {x.shape} and layout {(layout.seq_len, layout.d_model)}."
            )
        if layout.block_size != layout.d_model:
            raise NotImplementedError(
                "STIP attention currently requires block_size == d_model. "
                f"Got block_size={layout.block_size}, d_model={layout.d_model}."
            )
        if layout.total_slots_needed > x._cipher.size:
            raise RuntimeError(
                "STIP attention layout exceeds ciphertext capacity. "
                f"Need {layout.total_slots_needed} slots, have {x._cipher.size}."
            )
        if layout.seq_len > 8:
            raise NotImplementedError(
                f"seq_len={layout.seq_len} not supported. Maximum is 8 for {self.normalization_mode} attention."
            )
        return layout.seq_len, layout.d_model

    def _forward_attention_stip(self, x: "EncryptedTensor") -> "EncryptedTensor":
        seq_len, embed_dim = self._validate_stip_layout(x)
        layout = x._packing_layout
        assert layout is not None

        packed_x = x.clone()
        packed_x._packed_batch = True
        packed_x._batch_size = 1
        packed_x._slots_per_sample = seq_len * embed_dim
        packed_x._packed_sample_shape = (seq_len, embed_dim)
        packed_x._stip_layout_fresh = False

        output = self._forward_attention_packed(packed_x)
        output._packed_batch = False
        output._batch_size = None
        output._slots_per_sample = None
        output._packed_sample_shape = None
        output._shape = (seq_len, embed_dim)
        output._packing_layout = layout
        output._stip_layout_fresh = True
        return output

    def _normalize_positive_scores(
        self,
        scores: List["EncryptedTensor"],
        eps: float = 0.01,
        reciprocal_domain: tuple[float, float] = (0.5, 150.0),
        renorm_reciprocal_domain: tuple[float, float] = (0.5, 10.0),
        renormalize: bool = True,
        reciprocal_fill: Optional[List[float]] = None,
        renorm_reciprocal_fill: Optional[List[float]] = None,
    ) -> List["EncryptedTensor"]:
        from cukks.stats.crypto_reciprocal import crypto_reciprocal_shallow

        z_sum = scores[0]
        for score in scores[1:]:
            z_sum = z_sum.add(score)

        z_sum_safe = z_sum.add(eps)
        if reciprocal_fill is not None:
            z_sum_safe = z_sum_safe.add(reciprocal_fill)
        inv_z_sum = crypto_reciprocal_shallow(z_sum_safe, domain=reciprocal_domain)
        weights = [score.mul(inv_z_sum).rescale() for score in scores]

        if not renormalize:
            return weights

        weight_sum = weights[0]
        for weight in weights[1:]:
            weight_sum = weight_sum.add(weight)

        weight_sum_safe = weight_sum.add(eps)
        if renorm_reciprocal_fill is not None:
            weight_sum_safe = weight_sum_safe.add(renorm_reciprocal_fill)
        inv_weight_sum = crypto_reciprocal_shallow(weight_sum_safe, domain=renorm_reciprocal_domain)
        return [weight.mul(inv_weight_sum).rescale() for weight in weights]

    def _power_reciprocal_domain(self, num_scores: int, shift: float) -> tuple[float, float]:
        margin = 0.5
        lower = max(0.5, float(num_scores) * max(shift - margin, 0.25) ** 2)
        upper = max(lower + 1.0, float(num_scores) * (shift + margin) ** 2)
        return (lower, upper)

    def _power_renorm_reciprocal_domain(self) -> tuple[float, float]:
        return (0.75, 1.25)

    def _power_reciprocal_coeffs_for(self, num_scores: int, shift: float) -> list[float]:
        return _compute_reciprocal_coeffs(
            self._power_reciprocal_domain(num_scores, shift),
            degree=self._reciprocal_degree,
        )

    def _power_renorm_reciprocal_coeffs_for(self) -> list[float]:
        return _compute_reciprocal_coeffs(
            self._power_renorm_reciprocal_domain(),
            degree=self._reciprocal_degree,
        )

    def _gaussian_reciprocal_domain(self, num_scores: int) -> tuple[float, float]:
        lower = max(0.05, 0.5 * float(num_scores) * math.exp(-self._gaussian_domain[1]))
        upper = float(num_scores) + 0.5
        return (lower, upper)

    def _forward_attention_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        sample_shape = getattr(x, "_packed_sample_shape", None) or x.shape[1:]
        if len(sample_shape) == 1:
            if sample_shape[0] != self.embed_dim:
                raise RuntimeError(
                    f"Packed attention expected sample shape ({self.embed_dim},), got {sample_shape}."
                )

            q = self._apply_projection(x, self.q_weight, self.q_bias)
            k = self._apply_projection(x, self.k_weight, self.k_bias)
            v = self._apply_projection(x, self.v_weight, self.v_bias)
            scores = q.mul(k).rescale().sum_and_broadcast(self.embed_dim).mul(self.scale).rescale()
            attn_weights = self._approx_softmax_row(scores, seq_len=1)
            output = attn_weights.mul(v).rescale()
            output = self._apply_projection(output, self.out_weight, self.out_bias)
            return output

        if len(sample_shape) != 2 or sample_shape[1] != self.embed_dim:
            raise RuntimeError(
                "Packed attention requires packed sample shape (seq_len, embed_dim), got "
                f"{sample_shape}."
            )

        seq_len = sample_shape[0]
        embed_dim = sample_shape[1]
        if seq_len > 8:
            raise NotImplementedError(
                f"seq_len={seq_len} not supported. Maximum is 8 for {self.normalization_mode} attention."
            )
        query_masks, power_outside_fills, gaussian_outside_fills, rotation_offsets, ones_block = self._precompute_packed_metadata(
            x,
            seq_len,
            embed_dim,
        )

        q = self._apply_projection(x, self.q_weight, self.q_bias)
        k = self._apply_projection(x, self.k_weight, self.k_bias)
        v = self._apply_projection(x, self.v_weight, self.v_bias)
        if (
            self.normalization_mode == "power_softmax"
            and hasattr(q._cipher, "packed_self_attention_power")
            and not hasattr(q._cipher, "_backend")
        ):
            combined = q.packed_self_attention_power(
                k,
                v,
                batch_size=getattr(x, "_batch_size", 1),
                seq_len=seq_len,
                embed_dim=embed_dim,
                scale=self.scale,
                shift=2.0,
                reciprocal_coeffs=self._power_reciprocal_coeffs_for(seq_len, shift=2.0),
                renorm_reciprocal_coeffs=self._power_renorm_reciprocal_coeffs_for(),
            )
            combined = combined.rescale()
            combined = self._apply_projection(combined, self.out_weight, self.out_bias)
            return combined.view(*x.shape)
        rotated_k = {0: k}
        rotated_v = {0: v}
        for offset in rotation_offsets:
            if offset == 0:
                continue
            rotated_k[offset] = k.rotate(offset)
            rotated_v[offset] = v.rotate(offset)

        outputs = []
        for query_index in range(seq_len):
            query_mask = query_masks[query_index]
            reciprocal_fill = [1.0 - value for value in query_mask]
            power_outside_fill = power_outside_fills[query_index]
            gaussian_outside_fill = gaussian_outside_fills[query_index]
            score_terms = []
            for key_index in range(seq_len):
                offset = (key_index - query_index) * embed_dim
                if self.normalization_mode == "gaussian":
                    diff = q.sub(rotated_k[offset])
                    squared_diff = diff.mul(diff).rescale()
                    masked_scores = squared_diff.mul(query_mask).rescale()
                    score = masked_scores.view(getattr(x, "_batch_size", 1), seq_len, embed_dim).matmul(ones_block)
                    score = score.mul(self.scale).rescale()
                    score_terms.append(score.add(gaussian_outside_fill))
                else:
                    aligned_product = q.mul(rotated_k[offset]).rescale()
                    masked_scores = aligned_product.mul(query_mask).rescale()
                    score = masked_scores.view(getattr(x, "_batch_size", 1), seq_len, embed_dim).matmul(ones_block)
                    score = score.mul(self.scale).rescale()
                    score = score.add(power_outside_fill)
                    score_terms.append(score)

            weights = self._compute_attention_weights(score_terms, reciprocal_fill=reciprocal_fill)
            token_output = None
            for key_index, weight in enumerate(weights):
                offset = (key_index - query_index) * embed_dim
                term = weight.mul(rotated_v[offset]).rescale()
                token_output = term if token_output is None else token_output.add(term)

            assert token_output is not None
            token_output = token_output.mul(query_mask).rescale()
            outputs.append(token_output)

        combined = outputs[0]
        for token_output in outputs[1:]:
            combined = combined.add(token_output)

        combined = self._apply_projection(combined, self.out_weight, self.out_bias)
        return combined.view(*x.shape)

    def _ct_dot(
        self, a: "EncryptedTensor", b: "EncryptedTensor", dim: int
    ) -> "EncryptedTensor":
        """Compute cipher-cipher dot product: sum(a * b)."""
        product = a.mul(b).rescale()
        return product.sum_and_broadcast(dim)

    def _pcmm(
        self,
        columns: list["EncryptedTensor"],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> list["EncryptedTensor"]:
        """Plaintext-Ciphertext Matrix Multiply (column-wise)."""
        d_in = len(columns)
        w = weight.to(dtype=torch.float64)
        if w.ndim != 2:
            raise ValueError(f"Expected rank-2 weight matrix, got shape {tuple(w.shape)}")
        if w.shape[0] != d_in and w.shape[1] != d_in:
            raise ValueError(
                "Weight is incompatible with input columns: "
                f"d_in={d_in}, weight_shape={tuple(w.shape)}"
            )
        if w.shape[0] != d_in:
            w = w.T

        out_cols: list["EncryptedTensor"] = []
        for out_idx in range(w.shape[1]):
            acc: Optional["EncryptedTensor"] = None
            for in_idx in range(d_in):
                coeff = float(w[in_idx, out_idx])
                if abs(coeff) < 1e-30:
                    continue
                term = columns[in_idx].mul(coeff)
                acc = term if acc is None else acc.add(term)

            if acc is None:
                acc = columns[0].mul(0.0)
            acc = acc.rescale()
            if bias is not None:
                acc = acc.add(float(bias[out_idx]))
            out_cols.append(acc)
        return out_cols

    def _diagonal_softmax(self, score_diagonals: list["EncryptedTensor"]) -> list["EncryptedTensor"]:
        """Apply configured attention normalization to diagonal scores."""
        return self._compute_attention_weights(score_diagonals)

    def _head_attention_from_columns(
        self,
        q_head: list["EncryptedTensor"],
        k_head: list["EncryptedTensor"],
        v_head: list["EncryptedTensor"],
        seq_len: int,
        rihp: Optional[Any],
    ) -> list["EncryptedTensor"]:
        if rihp is not None:
            k_hybrid = rihp.pack_hybrid(k_head)
            hybrid_diags = rihp.halved_ccmm(q_head, k_hybrid)
            score_diags = rihp.unpack_diagonals(hybrid_diags)
        else:
            score_diags = []
            for rotation in range(seq_len):
                acc = q_head[0].mul(k_head[0].rotate(rotation)).rescale()
                for col_idx in range(1, self.head_dim):
                    term = q_head[col_idx].mul(k_head[col_idx].rotate(rotation)).rescale()
                    acc = acc.add(term)
                score_diags.append(acc)

        scaled_scores = [diag.mul(self.scale).rescale() for diag in score_diags]
        weights_diags = self._diagonal_softmax(scaled_scores)

        out_cols: list["EncryptedTensor"] = []
        for col_idx in range(self.head_dim):
            acc = weights_diags[0].mul(v_head[col_idx]).rescale()
            for rotation in range(1, seq_len):
                term = weights_diags[rotation].mul(v_head[col_idx].rotate(rotation)).rescale()
                acc = acc.add(term)
            out_cols.append(acc)
        return out_cols

    def forward_stip_columns_dhp(
        self,
        x_columns: list["EncryptedTensor"],
        seq_len: int,
    ) -> list["EncryptedTensor"]:
        """Column-wise STIP attention with DHP projection packing."""
        if self.num_heads % 2 != 0:
            raise ValueError("forward_stip_columns_dhp requires even num_heads")

        d_model = len(x_columns)
        if d_model != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} columns, got {d_model}")

        pairer = batching_module.DHPacker(self.num_heads)
        use_rihp = seq_len % 2 == 0
        rihp = batching_module.RIHPacker(seq_len) if use_rihp else None
        all_repacked: list["EncryptedTensor"] = []

        if self.q_weight is not None:
            if self.k_weight is None or self.v_weight is None:
                raise RuntimeError("q/k/v weights must all be set or all be None")

            q_w = self.q_weight.to(dtype=torch.float64)
            k_w = self.k_weight.to(dtype=torch.float64)
            v_w = self.v_weight.to(dtype=torch.float64)
            if q_w.shape[0] != d_model:
                q_w = q_w.T
            if k_w.shape[0] != d_model:
                k_w = k_w.T
            if v_w.shape[0] != d_model:
                v_w = v_w.T

            for pair_idx in range(pairer.num_pairs):
                left_start = pair_idx * 2 * self.head_dim
                left_end = left_start + self.head_dim
                right_start = left_end
                right_end = right_start + self.head_dim

                q_precoded = batching_module.DHPacker.precode_weights(q_w[:, left_start:left_end], q_w[:, right_start:right_end])
                k_precoded = batching_module.DHPacker.precode_weights(k_w[:, left_start:left_end], k_w[:, right_start:right_end])
                v_precoded = batching_module.DHPacker.precode_weights(v_w[:, left_start:left_end], v_w[:, right_start:right_end])

                q_packed = pairer.parallel_projection(x_columns, q_precoded.tolist())
                k_packed = pairer.parallel_projection(x_columns, k_precoded.tolist())
                v_packed = pairer.parallel_projection(x_columns, v_precoded.tolist())

                q_left, q_right = pairer.unpack_heads(q_packed)
                k_left, k_right = pairer.unpack_heads(k_packed)
                v_left, v_right = pairer.unpack_heads(v_packed)

                for col_idx in range(self.head_dim):
                    if self.q_bias is not None:
                        q_left[col_idx] = q_left[col_idx].add(float(self.q_bias[left_start + col_idx]))
                        q_right[col_idx] = q_right[col_idx].add(float(self.q_bias[right_start + col_idx]))
                    if self.k_bias is not None:
                        k_left[col_idx] = k_left[col_idx].add(float(self.k_bias[left_start + col_idx]))
                        k_right[col_idx] = k_right[col_idx].add(float(self.k_bias[right_start + col_idx]))
                    if self.v_bias is not None:
                        v_left[col_idx] = v_left[col_idx].add(float(self.v_bias[left_start + col_idx]))
                        v_right[col_idx] = v_right[col_idx].add(float(self.v_bias[right_start + col_idx]))

                out_left = self._head_attention_from_columns(q_left, k_left, v_left, seq_len, rihp)
                out_right = self._head_attention_from_columns(q_right, k_right, v_right, seq_len, rihp)
                repacked = pairer.repack_after_attention(out_left, out_right)

                all_repacked.extend(repacked)
        else:
            for pair_idx in range(pairer.num_pairs):
                left_start = pair_idx * 2 * self.head_dim
                left_end = left_start + self.head_dim
                right_start = left_end
                right_end = right_start + self.head_dim

                out_left = self._head_attention_from_columns(
                    x_columns[left_start:left_end],
                    x_columns[left_start:left_end],
                    x_columns[left_start:left_end],
                    seq_len,
                    rihp,
                )
                out_right = self._head_attention_from_columns(
                    x_columns[right_start:right_end],
                    x_columns[right_start:right_end],
                    x_columns[right_start:right_end],
                    seq_len,
                    rihp,
                )
                repacked = pairer.repack_after_attention(out_left, out_right)
                all_repacked.extend(repacked)

        if self.out_weight is None:
            out_cols: list["EncryptedTensor"] = []
            for pair_idx in range(pairer.num_pairs):
                pair_start = pair_idx * self.head_dim
                pair_end = pair_start + self.head_dim
                left, right = pairer.unpack_heads(all_repacked[pair_start:pair_end])
                out_cols.extend(left)
                out_cols.extend(right)
            return out_cols

        out_w = self.out_weight.to(dtype=torch.float64)
        if out_w.shape[0] != self.embed_dim:
            out_w = out_w.T

        final_outputs: list["EncryptedTensor"] = []
        for out_idx in range(self.embed_dim):
            pair_accum: Optional["EncryptedTensor"] = None
            for pair_idx in range(pairer.num_pairs):
                left_start = pair_idx * 2 * self.head_dim
                left_end = left_start + self.head_dim
                right_start = left_end
                right_end = right_start + self.head_dim

                packed_pair = all_repacked[pair_idx * self.head_dim : (pair_idx + 1) * self.head_dim]
                left_weights = out_w[left_start:left_end, out_idx]
                right_weights = out_w[right_start:right_end, out_idx]
                precoded = batching_module.DHPacker.precode_weights(left_weights.unsqueeze(1), -right_weights.unsqueeze(1))
                pair_proj = pairer.parallel_projection(packed_pair, precoded.tolist())
                if not pair_proj:
                    continue
                pair_term = pair_proj[0]
                pair_accum = pair_term if pair_accum is None else pair_accum.add(pair_term)

            if pair_accum is None:
                pair_accum = x_columns[0].mul(0.0)
            final_col = pairer.extract_final(pair_accum).rescale()
            if self.out_bias is not None:
                final_col = final_col.add(float(self.out_bias[out_idx]))
            final_outputs.append(final_col)

        return final_outputs

    def forward_stip_columns(
        self,
        x_columns: list["EncryptedTensor"],
        seq_len: int,
    ) -> list["EncryptedTensor"]:
        """Column-wise STIP attention using RIHP + optional DHP projection packing."""
        d_model = len(x_columns)
        if d_model != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} columns, got {d_model}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > 8:
            raise NotImplementedError(
                f"seq_len={seq_len} not supported. Maximum is 8 for {self.normalization_mode} attention."
            )

        if self.num_heads % 2 == 0:
            return self.forward_stip_columns_dhp(x_columns, seq_len)

        if self.q_weight is not None:
            if self.k_weight is None or self.v_weight is None:
                raise RuntimeError("q/k/v weights must all be set or all be None")
            q_cols = self._pcmm(x_columns, self.q_weight, self.q_bias)
            k_cols = self._pcmm(x_columns, self.k_weight, self.k_bias)
            v_cols = self._pcmm(x_columns, self.v_weight, self.v_bias)
        else:
            q_cols = list(x_columns)
            k_cols = list(x_columns)
            v_cols = list(x_columns)

        rihp = batching_module.RIHPacker(seq_len) if seq_len % 2 == 0 else None
        all_out_cols: list["EncryptedTensor"] = []
        for head_idx in range(self.num_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim
            all_out_cols.extend(
                self._head_attention_from_columns(
                    q_cols[start:end],
                    k_cols[start:end],
                    v_cols[start:end],
                    seq_len,
                    rihp,
                )
            )

        if self.out_weight is not None:
            all_out_cols = self._pcmm(all_out_cols, self.out_weight, self.out_bias)
        return all_out_cols

    def _power_softmax(
        self,
        scores: List["EncryptedTensor"],
        shift: float = 2.0,
        eps: float = 0.01,
        reciprocal_fill: Optional[List[float]] = None,
    ) -> List["EncryptedTensor"]:
        """Compute Power-Softmax (p=2) for attention weights.

        t_j = s_j + shift
        u_j = t_j^2
        Z = sum(u_j)
        w_j = u_j / Z
        """
        shifted = [s.add(shift) for s in scores]
        # Ensure shifted scores are rescaled before squaring.
        # If scores came from a mul (degree 2), squaring without rescale
        # would produce degree 4, causing rapid noise growth or decryption failure.
        shifted = [t._ensure_rescaled() for t in shifted]
        squared = [t.mul(t).rescale() for t in shifted]
        return self._normalize_positive_scores(
            squared,
            eps=eps,
            reciprocal_domain=self._power_reciprocal_domain(len(scores), shift),
            renorm_reciprocal_domain=self._power_renorm_reciprocal_domain(),
            renormalize=False,
            reciprocal_fill=reciprocal_fill,
        )

    def _gaussian_kernel_weights(
        self,
        distances: List["EncryptedTensor"],
        eps: float = 0.01,
        reciprocal_fill: Optional[List[float]] = None,
    ) -> List["EncryptedTensor"]:
        scaled_distances = [distance.mul(self.gaussian_gamma).rescale() for distance in distances]
        kernels = [self._approx_gaussian_kernel(distance) for distance in scaled_distances]
        return self._normalize_positive_scores(
            kernels,
            eps=eps,
            reciprocal_domain=self._gaussian_reciprocal_domain(len(distances)),
            renormalize=False,
            reciprocal_fill=reciprocal_fill,
        )

    def _approx_gaussian_kernel(self, distance: "EncryptedTensor") -> "EncryptedTensor":
        a, b = self._gaussian_domain
        alpha = 2.0 / (b - a)
        beta = -(a + b) / (b - a)
        t = distance.mul(alpha).rescale().add(beta)
        return t.poly_eval(self._gaussian_exp_coeffs)

    def _compute_attention_weights(
        self,
        score_terms: List["EncryptedTensor"],
        reciprocal_fill: Optional[List[float]] = None,
    ) -> List["EncryptedTensor"]:
        if self.normalization_mode == "gaussian":
            return self._gaussian_kernel_weights(score_terms, reciprocal_fill=reciprocal_fill)
        return self._power_softmax(score_terms, reciprocal_fill=reciprocal_fill)
    
    def _approx_exp(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Approximate exp(x) using Taylor series.
        
        exp(x) ≈ 1 + x + x²/2! + x³/3! + ...
        
        Args:
            x: Input encrypted tensor.
            
        Returns:
            Approximation of exp(x).
        """
        return x.poly_eval(self._exp_coeffs)
    
    def _approx_softmax_row(
        self,
        scores: "EncryptedTensor",
        seq_len: int,
    ) -> "EncryptedTensor":
        """Approximate softmax over the last dimension."""
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len == 1:
            return scores.mul(0.0).add(1.0)
        
        exp_scores = self._approx_exp(scores)
        norm_factor = 1.0 / seq_len
        
        return exp_scores.mul(norm_factor).rescale()
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Self-attention forward pass.
        
        For self-attention, uses ``x`` as query, key, and value.

        Single encrypted tensors support ``seq_len=1``. Packed encrypted
        batches with sample shape ``(seq_len, embed_dim)`` support
        multi-token attention up to ``seq_len=8``.
        """
        if getattr(x, "_packing_layout", None) is not None:
            return self._forward_attention_stip(x)
        if getattr(x, "_packed_batch", False):
            return self._forward_attention_packed(x)
        return cast("EncryptedTensor", self.forward_attention(x, x, x))
    
    def _forward_attention_multi(
        self,
        query: List["EncryptedTensor"],
        key: List["EncryptedTensor"],
        value: List["EncryptedTensor"],
    ) -> List["EncryptedTensor"]:
        seq_len = len(query)
        
        if seq_len == 0:
            return []

        if seq_len > 8:
            raise NotImplementedError(
                f"seq_len={seq_len} not supported. Maximum is 8 for {self.normalization_mode} attention."
            )
         
        if seq_len != len(key) or seq_len != len(value):
            raise ValueError("query, key, value must have same length.")
        
        q_list = [self._apply_projection(q, self.q_weight, self.q_bias) for q in query]
        k_list = [self._apply_projection(k, self.k_weight, self.k_bias) for k in key]
        v_list = [self._apply_projection(v, self.v_weight, self.v_bias) for v in value]
        
        outputs = []
        for i in range(seq_len):
            score_terms = []
            for j in range(seq_len):
                if self.normalization_mode == "gaussian":
                    diff = q_list[i].sub(k_list[j])
                    dist_ij = self._ct_dot(diff, diff, self.embed_dim)
                    dist_ij = dist_ij.mul(self.scale).rescale()
                    score_terms.append(dist_ij)
                else:
                    s_ij = self._ct_dot(q_list[i], k_list[j], self.embed_dim)
                    s_ij = s_ij.mul(self.scale).rescale()
                    score_terms.append(s_ij)
            
            weights = self._compute_attention_weights(score_terms)
            
            y_i = weights[0].mul(v_list[0]).rescale()
            for j in range(1, seq_len):
                y_i = y_i.add(weights[j].mul(v_list[j]).rescale())
            
            y_i = self._apply_projection(y_i, self.out_weight, self.out_bias)
            outputs.append(y_i)
        
        return outputs

    def forward_attention(
        self,
        query: Union["EncryptedTensor", List["EncryptedTensor"]],
        key: Union["EncryptedTensor", List["EncryptedTensor"]],
        value: Union["EncryptedTensor", List["EncryptedTensor"]],
    ) -> Union["EncryptedTensor", List["EncryptedTensor"]]:
        """Forward pass of approximate attention using pure HE operations.
        
        Supports two public input forms:

        - Single encrypted tensors for ``seq_len=1``
        - Lists of encrypted tensors for multi-token attention with
          ``seq_len <= 8`` using the configured normalization mode

        All computation stays in HE with no decryption fallback.
        
        Args:
            query: Query tensor for ``seq_len=1`` or a list of encrypted
                per-token tensors for multi-token attention.
            key: Key tensor/list matching ``query``.
            value: Value tensor/list matching ``query``.
            
        Returns:
            Encrypted tensor for single-token input or list of encrypted
            tensors for multi-token input.
            
        Raises:
            NotImplementedError: If a single encrypted tensor is passed with
                ``seq_len > 1`` or if list-based multi-token input exceeds
                ``seq_len=8``.
        """
        if isinstance(query, list):
            if not isinstance(key, list) or not isinstance(value, list):
                raise ValueError("query, key, value must have same length.")
            return self._forward_attention_multi(query, key, value)
        if isinstance(key, list) or isinstance(value, list):
            raise ValueError("query, key, value must have same length.")
        
        # Guard: only seq_len=1 supported
        if len(query.shape) > 1 and query.shape[0] > 1:
            raise NotImplementedError(
                "Single-tensor attention requires seq_len=1. For seq_len>1, "
                "use packed encrypted batches in forward() or pass query/key/value "
                "as List[EncryptedTensor] with max seq_len=8. "
                f"Got query shape {query.shape} with seq_len={query.shape[0]}."
            )
        
        # Apply input projections if weights are set
        q = self._apply_projection(query, self.q_weight, self.q_bias)
        k = self._apply_projection(key, self.k_weight, self.k_bias)
        v = self._apply_projection(value, self.v_weight, self.v_bias)
        
        # Q · K (element-wise cipher×cipher, then sum) = dot product
        # For seq_len=1: Q and K are both shape (embed_dim,)
        # Q @ K^T for 1D = sum(Q * K) which is a scalar score
        qk = q.mul(k).rescale()
        scores = qk.sum_and_broadcast(self.embed_dim).mul(self.scale).rescale()
        
        attn_weights = self._approx_softmax_row(scores, seq_len=1)
        
        # attn_weights · V (element-wise cipher×cipher)
        # For seq_len=1: attn_weight is effectively a scalar broadcast to embed_dim slots
        # Multiplying with V gives the weighted value
        output = attn_weights.mul(v).rescale()
        
        # Apply output projection if set
        output = self._apply_projection(output, self.out_weight, self.out_bias)
        
        return output
    
    def mult_depth(self) -> int:
        """Estimate multiplicative depth of attention.
        
        For seq_len=1 (Taylor softmax):
        - Q/K/V projections: 3 (if used)
        - Q * K (cipher×cipher): 1
        - Softmax polynomial: softmax_degree
        - attn * V (cipher×cipher): 1
        - Output projection: 1 (if used)
        
        For seq_len>1 (Power-Softmax):
        - Q/K/V projections: 3 (if used)
        - Q * K dot products: 1 per pair
        - Power-Softmax: square(1) + reciprocal(~4) + weight_mul(1) ≈ 6
        - Weighted V sum: 1
        - Output projection: 1 (if used)
        
        Returns depth for Power-Softmax case (~8-12 for typical configs).
        """
        depth = 0
        
        # Input projections (parallel: Q, K, V all computed at same depth)
        if self.q_weight is not None:
            depth += 1  # max of Q, K, V projections (parallel)
        
        # Q * K cipher×cipher
        depth += 1

        if self.normalization_mode == "gaussian":
            gaussian_poly_depth = max(1, math.ceil(math.log2(self.softmax_degree + 1)))
            depth += 1 + gaussian_poly_depth + 4
        else:
            # Power-Softmax: square + reciprocal + weight multiplication
            # square: 1, reciprocal: ~4, weight_mul: 1
            depth += 6
        
        # Weighted V sum
        depth += 1
        
        # Output projection
        if self.out_weight is not None:
            depth += 1
        
        return depth
    
    @classmethod
    def from_torch(
        cls,
        attention: torch.nn.MultiheadAttention,
        softmax_degree: int = 4,
        normalization_mode: str = "power_softmax",
        gaussian_gamma: float = 0.25,
    ) -> "EncryptedApproxAttention":
        """Create from PyTorch MultiheadAttention.
        
        Args:
            attention: PyTorch MultiheadAttention module.
            softmax_degree: Degree for softmax polynomial approximation.
            normalization_mode: Multi-token normalization kernel.
            gaussian_gamma: Scale factor for gaussian attention mode.
            
        Returns:
            EncryptedApproxAttention with copied weights.
        """
        embed_dim = attention.embed_dim
        num_heads = attention.num_heads
        
        enc_attn = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            softmax_degree=softmax_degree,
            normalization_mode=normalization_mode,
            gaussian_gamma=gaussian_gamma,
        )
        
        # Extract weights from PyTorch attention
        # MultiheadAttention stores in_proj_weight as (3*embed_dim, embed_dim)
        # containing [W_q, W_k, W_v] stacked
        if attention.in_proj_weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
            in_proj = attention.in_proj_weight.data.detach().to(dtype=torch.float64)
            enc_attn.q_weight = in_proj[:embed_dim, :]
            enc_attn.k_weight = in_proj[embed_dim:2*embed_dim, :]
            enc_attn.v_weight = in_proj[2*embed_dim:, :]
        elif attention.q_proj_weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
            enc_attn.q_weight = attention.q_proj_weight.data.detach().to(dtype=torch.float64)
            enc_attn.k_weight = attention.k_proj_weight.data.detach().to(dtype=torch.float64)  # pyright: ignore[reportOptionalMemberAccess]
            enc_attn.v_weight = attention.v_proj_weight.data.detach().to(dtype=torch.float64)  # pyright: ignore[reportOptionalMemberAccess]
        
        if attention.in_proj_bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            in_bias = attention.in_proj_bias.data.detach().to(dtype=torch.float64)
            enc_attn.q_bias = in_bias[:embed_dim]
            enc_attn.k_bias = in_bias[embed_dim:2*embed_dim]
            enc_attn.v_bias = in_bias[2*embed_dim:]
        
        if attention.out_proj is not None:  # pyright: ignore[reportUnnecessaryComparison]
            enc_attn.out_weight = attention.out_proj.weight.data.detach().to(dtype=torch.float64)
            if attention.out_proj.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                enc_attn.out_bias = attention.out_proj.bias.data.detach().to(dtype=torch.float64)
        
        return enc_attn
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"softmax_degree={self.softmax_degree}, "
            f"normalization_mode={self.normalization_mode}, "
            f"gaussian_gamma={self.gaussian_gamma}"
        )
