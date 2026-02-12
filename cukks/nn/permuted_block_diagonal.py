"""
Permutation-based block-diagonalization of pre-trained dense weights.

Given a pre-trained model with dense linear layers, find row/column
permutations that maximize the energy in the block-diagonal structure,
then absorb those permutations into adjacent layers for free.

Key insight: for element-wise activations (x², ReLU, etc.),
    act(P @ x) = P @ act(x)
so permutations pass through activations and can be folded into
adjacent dense weight matrices without any CKKS overhead.

Usage::

    model = load_pretrained_model()
    decomposed = decompose_with_permutation(
        model,
        target_layer="fc2",
        prev_layer="fc1",
        next_layer="fc3",
        block_size=32,
    )
    # decomposed model now has fc2 as BlockDiagonalLinear,
    # with fc1 and fc3 modified to absorb the permutations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


def _spectral_permutation(
    W: torch.Tensor,
    num_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find row and column permutations that maximize block-diagonal energy.

    Uses spectral co-clustering on the absolute weight matrix to group
    rows and columns that interact strongly into the same block.

    Parameters
    ----------
    W : Tensor (out_features, in_features)
        Dense weight matrix.
    num_blocks : int
        Number of diagonal blocks desired.

    Returns
    -------
    row_perm, col_perm : LongTensor
        Permutation indices such that ``W[row_perm][:, col_perm]`` is
        approximately block-diagonal.
    """
    n_out, n_in = W.shape
    assert n_out == n_in, "Only square matrices supported"
    n = n_out
    block_size = n // num_blocks

    A = W.detach().cpu().float().abs().numpy()

    # Spectral biclustering via SVD of the absolute weight matrix.
    # The leading singular vectors reveal the block structure.
    # We use the approach from Dhillon (2001) spectral co-clustering.
    #
    # 1. Normalize A → D_r^{-1/2} A D_c^{-1/2}
    # 2. SVD → use top-k left/right singular vectors
    # 3. k-means on stacked [U; V] rows to get biclusters

    # Row and column sums for normalization (add epsilon for stability)
    eps = 1e-10
    d_r = A.sum(axis=1) + eps  # (n,)
    d_c = A.sum(axis=0) + eps  # (n,)

    D_r_inv_sqrt = np.diag(1.0 / np.sqrt(d_r))
    D_c_inv_sqrt = np.diag(1.0 / np.sqrt(d_c))

    A_norm = D_r_inv_sqrt @ A @ D_c_inv_sqrt

    # SVD — take top num_blocks singular vectors (skip the trivial first one)
    k = min(num_blocks, n - 1)
    U_full, _S, Vt_full = np.linalg.svd(A_norm, full_matrices=False)

    # Use singular vectors 1..k (skip vector 0 which is ~constant)
    # For co-clustering, stack scaled left and right vectors
    U_k = U_full[:, 1:k+1]  # (n, k)
    V_k = Vt_full[1:k+1, :].T  # (n, k)

    # Scale by D^{-1/2} to undo normalization
    U_k = D_r_inv_sqrt @ U_k
    V_k = D_c_inv_sqrt @ V_k

    # K-means clustering on row and column features separately
    from sklearn.cluster import KMeans

    row_kmeans = KMeans(n_clusters=num_blocks, n_init=20, random_state=42)
    row_labels = row_kmeans.fit_predict(U_k)

    col_kmeans = KMeans(n_clusters=num_blocks, n_init=20, random_state=42)
    col_labels = col_kmeans.fit_predict(V_k)

    # Build permutations by sorting indices within each cluster
    # Also match row clusters to column clusters for maximum alignment
    row_perm = _build_balanced_permutation(row_labels, num_blocks, block_size)
    col_perm = _build_balanced_permutation(col_labels, num_blocks, block_size)

    # Match row clusters to column clusters to maximize diagonal energy
    row_perm, col_perm = _match_row_col_clusters(
        W.detach().cpu().float(), row_perm, col_perm, num_blocks, block_size
    )

    return (
        torch.tensor(row_perm, dtype=torch.long),
        torch.tensor(col_perm, dtype=torch.long),
    )


def _build_balanced_permutation(
    labels: np.ndarray,
    num_blocks: int,
    block_size: int,
) -> np.ndarray:
    """Build a permutation from cluster labels, enforcing balanced block sizes.

    If clusters are unbalanced, redistributes excess elements to smaller
    clusters based on distance to cluster centroids (greedy).
    """
    target_size = block_size

    # Group indices by cluster
    clusters = {k: [] for k in range(num_blocks)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Balance: move excess elements to underfull clusters
    overflow = []
    for k in range(num_blocks):
        if len(clusters[k]) > target_size:
            # Keep the first target_size, move rest to overflow
            overflow.extend(clusters[k][target_size:])
            clusters[k] = clusters[k][:target_size]

    # Assign overflow to underfull clusters
    for k in range(num_blocks):
        deficit = target_size - len(clusters[k])
        if deficit > 0 and overflow:
            take = min(deficit, len(overflow))
            clusters[k].extend(overflow[:take])
            overflow = overflow[take:]

    # If still overflow (shouldn't happen for exact division), append to last
    if overflow:
        clusters[num_blocks - 1].extend(overflow)

    # Build permutation: block 0 indices, block 1 indices, ...
    perm = []
    for k in range(num_blocks):
        perm.extend(clusters[k])

    return np.array(perm, dtype=np.int64)


def _match_row_col_clusters(
    W: torch.Tensor,
    row_perm: np.ndarray,
    col_perm: np.ndarray,
    num_blocks: int,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-order row blocks to maximize alignment with column blocks.

    Tries all permutations of row blocks (for small num_blocks) or uses
    a greedy matching (for large num_blocks) to maximize the total
    block-diagonal Frobenius norm.
    """
    W_permuted_cols = W[:, col_perm].numpy()  # only permute columns first

    if num_blocks <= 8:
        # Brute-force: try all permutations of row block ordering
        from itertools import permutations as iter_perms

        best_energy = -1.0
        best_row_block_order = list(range(num_blocks))

        for perm_order in iter_perms(range(num_blocks)):
            # Build new row permutation with this block ordering
            new_row_perm = []
            for block_idx in perm_order:
                start = block_idx * block_size
                new_row_perm.extend(row_perm[start:start + block_size])
            new_row_perm = np.array(new_row_perm, dtype=np.int64)

            # Compute block-diagonal energy
            Wp = W_permuted_cols[new_row_perm]
            energy = 0.0
            for b in range(num_blocks):
                r = b * block_size
                block = Wp[r:r+block_size, r:r+block_size]
                energy += float(np.sum(block ** 2))

            if energy > best_energy:
                best_energy = energy
                best_row_block_order = list(perm_order)

        # Rebuild row_perm with best ordering
        new_row_perm = []
        for block_idx in best_row_block_order:
            start = block_idx * block_size
            new_row_perm.extend(row_perm[start:start + block_size])
        row_perm = np.array(new_row_perm, dtype=np.int64)

    else:
        # Greedy matching for large num_blocks
        # Compute energy matrix: E[i,j] = ||W[row_block_i, col_block_j]||_F^2
        E = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            ri = row_perm[i*block_size:(i+1)*block_size]
            for j in range(num_blocks):
                cj = col_perm[j*block_size:(j+1)*block_size]
                block = W.numpy()[np.ix_(ri, cj)]
                E[i, j] = float(np.sum(block ** 2))

        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        # Maximize → negate for minimization
        row_ind, col_ind = linear_sum_assignment(-E)

        # col_ind[i] = which column block should row block i map to
        # We want row block i to sit at position col_ind[i]
        new_row_perm = np.empty_like(row_perm)
        for i, j in zip(row_ind, col_ind):
            src = row_perm[i*block_size:(i+1)*block_size]
            new_row_perm[j*block_size:(j+1)*block_size] = src
        row_perm = new_row_perm

    return row_perm, col_perm


def compute_block_diagonal_energy(
    W: torch.Tensor,
    block_size: int,
    row_perm: Optional[torch.Tensor] = None,
    col_perm: Optional[torch.Tensor] = None,
) -> float:
    """Compute fraction of Frobenius energy in block-diagonal structure.

    Returns a value in [0, 1] where 1 means the matrix is perfectly
    block-diagonal after permutation.
    """
    if row_perm is not None:
        W = W[row_perm]
    if col_perm is not None:
        W = W[:, col_perm]

    n = W.shape[0]
    num_blocks = n // block_size
    total_energy = float((W ** 2).sum())
    if total_energy == 0:
        return 1.0

    bd_energy = 0.0
    for b in range(num_blocks):
        r = b * block_size
        block = W[r:r+block_size, r:r+block_size]
        bd_energy += float((block ** 2).sum())

    return bd_energy / total_energy


def decompose_with_permutation(
    model: nn.Module,
    target_layer: str,
    prev_layer: str,
    next_layer: str,
    block_size: int,
    *,
    calibration_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    refine_permutation: bool = False,
    refine_rounds: int = 3,
) -> nn.Module:
    """Decompose a dense layer into block-diagonal using optimal permutations.

    Modifies the model in-place:
    - ``target_layer`` becomes a ``BlockDiagonalLinear``
    - ``prev_layer`` weight is modified: ``W_prev' = Q @ W_prev``
    - ``next_layer`` weight is modified: ``W_next' = W_next @ P^T``
    - Biases are permuted accordingly.

    If ``calibration_data`` is provided, each block is optimized via
    per-block least-squares regression (GPTQ-style), which significantly
    improves accuracy preservation compared to naive weight slicing.

    Parameters
    ----------
    model : nn.Module
        The model containing the layers.
    target_layer : str
        Attribute name of the layer to block-diagonalize (e.g. "fc2").
    prev_layer : str
        Attribute name of the preceding dense layer (e.g. "fc1").
    next_layer : str
        Attribute name of the following dense layer (e.g. "fc3").
    block_size : int
        Block size for the block-diagonal structure.
    calibration_data : optional tuple (X_input, Y_output)
        Calibration activations at the target layer. ``X_input`` has shape
        ``(N, dim)`` (layer inputs), ``Y_output`` has shape ``(N, dim)``
        (layer outputs). When provided, per-block regression replaces
        naive weight slicing for much better accuracy.

    Returns
    -------
    model : nn.Module
        The modified model (same object, modified in-place).
    """
    from .block_diagonal import BlockDiagonalLinear

    # Get the target layer (could be nn.Linear or BlockDiagonalLinear)
    target = getattr(model, target_layer)
    if isinstance(target, BlockDiagonalLinear):
        W_target = target.to_dense_weight().detach().float()
    elif isinstance(target, nn.Linear):
        W_target = target.weight.detach().float()
    else:
        raise TypeError(f"Unsupported target layer type: {type(target)}")

    n = W_target.shape[0]
    assert W_target.shape[1] == n, "Target layer must be square"
    assert n % block_size == 0, f"Dimension {n} not divisible by block_size {block_size}"
    num_blocks = n // block_size

    row_perm, col_perm = _spectral_permutation(W_target, num_blocks)

    if refine_permutation:
        row_perm, col_perm = _refine_permutation_local_search(
            W_target, row_perm, col_perm, block_size, max_rounds=refine_rounds,
        )

    orig_energy = compute_block_diagonal_energy(W_target, block_size)
    perm_energy = compute_block_diagonal_energy(W_target, block_size, row_perm, col_perm)

    # Permute the target weight
    W_permuted = W_target[row_perm][:, col_perm]

    # Build block-diagonal weights
    blocks = torch.zeros(num_blocks, block_size, block_size)
    new_bias = torch.zeros(n)

    if calibration_data is not None:
        # Per-block least-squares regression (activation-aware)
        X_input, Y_output = calibration_data
        X_perm = X_input[:, col_perm]    # permuted inputs
        Y_perm = Y_output[:, row_perm]   # permuted desired outputs

        for b in range(num_blocks):
            r = b * block_size
            X_b = X_perm[:, r:r+block_size].float()   # (N, bs)
            Y_b = Y_perm[:, r:r+block_size].float()   # (N, bs)

            # Solve: X_b @ B^T ≈ Y_b  → B^T = lstsq(X_b, Y_b)
            sol = torch.linalg.lstsq(X_b, Y_b)
            blocks[b] = sol.solution  # (bs, bs) — already B^T (einsum convention)

            # Optimal bias: mean residual
            new_bias[r:r+block_size] = (Y_b - X_b @ sol.solution).mean(dim=0)
    else:
        # Naive weight slicing (no calibration data)
        for b in range(num_blocks):
            r = b * block_size
            block_w = W_permuted[r:r+block_size, r:r+block_size]
            blocks[b] = block_w.T  # transpose for einsum convention

        if target.bias is not None:
            new_bias = target.bias.detach().float()[row_perm]

    # Create BlockDiagonalLinear
    bd_layer = BlockDiagonalLinear(n, n, block_size, bias=True)
    with torch.no_grad():
        bd_layer.blocks.copy_(blocks)
        bd_layer.bias.copy_(new_bias)

    setattr(model, target_layer, bd_layer)

    # Modify previous layer: W_prev' = Q @ W_prev, b_prev' = Q @ b_prev
    prev = getattr(model, prev_layer)
    assert isinstance(prev, nn.Linear), f"prev_layer must be nn.Linear, got {type(prev)}"
    with torch.no_grad():
        prev.weight.copy_(prev.weight[col_perm])
        if prev.bias is not None:
            prev.bias.copy_(prev.bias[col_perm])

    # Modify next layer: W_next' = W_next @ P^T = W_next[:, row_perm]
    next_mod = getattr(model, next_layer)
    assert isinstance(next_mod, nn.Linear), f"next_layer must be nn.Linear, got {type(next_mod)}"
    with torch.no_grad():
        next_mod.weight.copy_(next_mod.weight[:, row_perm])

    # Store metadata for reference
    model._permutation_info = {  # type: ignore[attr-defined]
        "target_layer": target_layer,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "orig_bd_energy": round(orig_energy, 6),
        "perm_bd_energy": round(perm_energy, 6),
        "energy_improvement": round(perm_energy / max(orig_energy, 1e-12), 2),
        "used_calibration": calibration_data is not None,
        "row_perm": row_perm,
        "col_perm": col_perm,
    }

    return model


def collect_calibration_data(
    model: nn.Module,
    target_layer: str,
    prev_activation: str,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect calibration activations at a target layer for regression.

    Parameters
    ----------
    model : nn.Module
        The model (in eval mode).
    target_layer : str
        Attribute name of the layer to calibrate (e.g. "fc2").
    prev_activation : str
        Attribute name of the activation *before* the target layer
        (e.g. "act1"). We hook its output as the target layer's input.
    dataloader : DataLoader
        Calibration data loader.
    num_batches : int
        Number of batches to collect.

    Returns
    -------
    X_input, Y_output : Tensor, Tensor
        Inputs and outputs of the target layer, shape ``(N, dim)``.
    """
    model.eval()
    target = getattr(model, target_layer)
    prev_act = getattr(model, prev_activation)

    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    def hook_in(_m: nn.Module, _i: object, out: torch.Tensor) -> None:
        inputs.append(out.detach())

    def hook_out(_m: nn.Module, _i: object, out: torch.Tensor) -> None:
        outputs.append(out.detach())

    h1 = prev_act.register_forward_hook(hook_in)
    h2 = target.register_forward_hook(hook_out)

    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            model(data)
            if i + 1 >= num_batches:
                break

    h1.remove()
    h2.remove()

    return torch.cat(inputs, dim=0), torch.cat(outputs, dim=0)


def _refine_permutation_local_search(
    W: torch.Tensor,
    row_perm: torch.Tensor,
    col_perm: torch.Tensor,
    block_size: int,
    max_rounds: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refine permutation via greedy pairwise swaps to maximize BD energy."""
    n = W.shape[0]
    num_blocks = n // block_size

    rp = row_perm.clone()
    cp = col_perm.clone()

    def _bd_energy(rp_: torch.Tensor, cp_: torch.Tensor) -> float:
        Wp = W[rp_][:, cp_]
        e = 0.0
        for b in range(num_blocks):
            r = b * block_size
            e += float((Wp[r:r+block_size, r:r+block_size] ** 2).sum())
        return e

    best_energy = _bd_energy(rp, cp)

    for _ in range(max_rounds):
        improved = False
        for perm_arr in [rp, cp]:
            for b1 in range(num_blocks):
                for b2 in range(b1 + 1, num_blocks):
                    for i in range(block_size):
                        for j in range(block_size):
                            idx1 = b1 * block_size + i
                            idx2 = b2 * block_size + j
                            perm_arr[idx1], perm_arr[idx2] = perm_arr[idx2].item(), perm_arr[idx1].item()
                            e = _bd_energy(rp, cp)
                            if e > best_energy:
                                best_energy = e
                                improved = True
                            else:
                                perm_arr[idx1], perm_arr[idx2] = perm_arr[idx2].item(), perm_arr[idx1].item()
        if not improved:
            break

    return rp, cp


def compensate_next_layer(
    model: nn.Module,
    next_layer: str,
    next_activation: str,
    dataloader: torch.utils.data.DataLoader,
    teacher_model: nn.Module,
    num_batches: int = 10,
    ridge_alpha: float = 1e-4,
) -> None:
    """Re-solve the next layer (fc3) to compensate for BD approximation error.

    Collects activations from the decomposed model (student) at the input
    of ``next_layer``, and target logits from the ``teacher_model``,
    then solves ``fc3_new = argmin ||H_student @ W3^T + b3 - logits_teacher||``
    via ridge regression.

    Parameters
    ----------
    model : nn.Module
        The decomposed model (student), modified in-place.
    next_layer : str
        Attribute name of the layer to compensate (e.g. "fc3").
    next_activation : str
        Attribute name of the activation before ``next_layer`` (e.g. "act2").
    dataloader : DataLoader
        Calibration data.
    teacher_model : nn.Module
        Original dense model providing target outputs.
    num_batches : int
        Number of calibration batches.
    ridge_alpha : float
        Ridge regularization coefficient.
    """
    model.eval()
    teacher_model.eval()

    act_layer = getattr(model, next_activation)
    student_h: list[torch.Tensor] = []
    teacher_out: list[torch.Tensor] = []

    hook = act_layer.register_forward_hook(
        lambda _m, _i, out: student_h.append(out.detach())
    )

    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            model(data)
            teacher_out.append(teacher_model(data).detach())
            if i + 1 >= num_batches:
                break

    hook.remove()

    H = torch.cat(student_h, dim=0).float()   # (N, hidden)
    Y = torch.cat(teacher_out, dim=0).float()  # (N, 10)

    # Ridge regression: W3^T = (H^T H + αI)^{-1} H^T Y
    # Then bias = mean(Y - H @ W3^T)
    HtH = H.T @ H
    HtH.diagonal().add_(ridge_alpha * H.shape[0])
    HtY = H.T @ Y
    W3_T = torch.linalg.solve(HtH, HtY)  # (hidden, 10)

    bias_new = (Y - H @ W3_T).mean(dim=0)  # (10,)

    fc3 = getattr(model, next_layer)
    with torch.no_grad():
        fc3.weight.copy_(W3_T.T)  # nn.Linear: y = x @ W^T + b
        fc3.bias.copy_(bias_new)
