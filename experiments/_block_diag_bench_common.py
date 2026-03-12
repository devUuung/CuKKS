from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn as nn

from cukks.nn.block_diagonal import BlockDiagonalLinear


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class BlockDiagMNIST(nn.Module):
    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.act1 = SquareActivation()
        self.fc2 = BlockDiagonalLinear(hidden, hidden, block_size=block_size)
        self.act2 = SquareActivation()
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class BlockDiagMNISTConvertible(nn.Module):
    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = BlockDiagonalLinear(hidden, hidden, block_size=block_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def count_nonzero_diagonals(layer: BlockDiagonalLinear) -> int:
    dense_w = layer.to_dense_weight().detach()
    rows = torch.arange(dense_w.shape[0])
    count = 0
    for diagonal_idx in range(dense_w.shape[1]):
        diagonal = dense_w[rows, (rows + diagonal_idx) % dense_w.shape[1]]
        if diagonal.abs().max() > 0:
            count += 1
    return count


def estimate_bsgs_rotations(in_features: int, nonzero_diags: int) -> dict[str, int]:
    n1 = math.ceil(math.sqrt(in_features))
    n2 = math.ceil(in_features / n1)
    baby_step_rots = n1 - 1
    nonempty_giant_steps = 0

    for giant_step_index in range(n2):
        giant_step = giant_step_index * n1
        has_nonzero = False
        for baby_step_index in range(n1):
            diagonal_idx = giant_step + baby_step_index
            if diagonal_idx >= in_features:
                break
            if diagonal_idx < nonzero_diags or (in_features - diagonal_idx) <= nonzero_diags:
                has_nonzero = True
                break
        if has_nonzero:
            nonempty_giant_steps += 1

    giant_step_rots = max(0, nonempty_giant_steps - 1)
    return {
        "total_rotations": baby_step_rots + giant_step_rots,
        "total_evalmults": nonzero_diags,
        "nonzero_diagonals": nonzero_diags,
        "dense_diagonals": in_features,
    }


def default_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/workspace/ckks-torch/openfhe-gpu-public/build/lib:"
        "/workspace/ckks-torch/openfhe-gpu-public/build/_deps/rmm-build",
    )
    return env


def parse_result_json(stdout: str) -> dict[str, Any] | None:
    for line in stdout.strip().splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    return None
