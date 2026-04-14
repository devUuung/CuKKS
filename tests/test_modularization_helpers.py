from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch.nn as nn

from cukks.cnn_layout import CNNLayoutAnalyzer
from cukks.depth_estimation import estimate_model_depth
from cukks.module_registry import build_converter_registry
from cukks.rotation_planning import collect_rotation_requirements


def test_estimate_model_depth_matches_layernorm_modes() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))

    default_depth = estimate_model_depth(model)
    inverse_free_depth = estimate_model_depth(model, inverse_free_ln_names=frozenset({"1"}))

    assert default_depth == 19
    assert inverse_free_depth == 6


def test_collect_rotation_requirements_tracks_small_output_pack_shifts() -> None:
    model = nn.Sequential(nn.Linear(32, 4))

    requirements = collect_rotation_requirements(model)

    assert requirements.reduction_lengths == [32]
    assert requirements.pack_shifts == [1, 2, 3]


def test_cnn_layout_analyzer_builds_sparse_linear_layout() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 8),
    )
    analyzer = CNNLayoutAnalyzer(input_shape=(1, 8, 8))
    analyzer.analyze(model)

    layout = analyzer.compute_layout_from_linear(model[-1])

    assert layout is not None
    assert layout["num_patches"] == 16
    assert layout["patch_features"] == 4
    assert layout["sparse"] is True
    assert layout["total_slots"] == 8 * 8 * 4


def test_build_converter_registry_includes_activation_overrides() -> None:
    class StubConverter:
        def _convert_linear(self):
            pass

        def _convert_block_diagonal(self):
            pass

        def _convert_block_diag_low_rank(self):
            pass

        def _convert_conv1d(self):
            pass

        def _convert_conv2d(self):
            pass

        def _convert_conv_transpose2d(self):
            pass

        def _convert_avgpool2d(self):
            pass

        def _convert_adaptive_avgpool2d(self):
            pass

        def _convert_flatten(self):
            pass

        def _convert_sequential(self):
            pass

        def _convert_batchnorm1d(self):
            pass

        def _convert_batchnorm2d(self):
            pass

        def _convert_groupnorm(self):
            pass

        def _convert_instancenorm1d(self):
            pass

        def _convert_instancenorm2d(self):
            pass

        def _convert_dropout(self):
            pass

        def _convert_maxpool2d(self):
            pass

        def _convert_layernorm(self):
            pass

        def _convert_attention(self):
            pass

        def _convert_upsample(self):
            pass

        def _convert_embedding(self):
            pass

        def _convert_pixel_shuffle(self):
            pass

        def _convert_pixel_unshuffle(self):
            pass

        def _convert_zeropad2d(self):
            pass

        def _convert_constantpad2d(self):
            pass

        def _convert_reflectionpad2d(self):
            pass

        def _convert_replicationpad2d(self):
            pass

        def _convert_activation(self):
            pass

    registry = build_converter_registry(StubConverter(), {nn.ReLU: object})

    assert nn.Linear in registry
    assert registry[nn.ReLU].__name__ == "_convert_activation"


def test_tracked_artifact_script_passes_in_clean_temp_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "README.md").write_text("ok\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_tracked_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No tracked build artifacts detected." in result.stdout
