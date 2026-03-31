from importlib import import_module

import torch
import torch.nn as nn


EncryptedConv1d = import_module("cukks.nn.conv1d").EncryptedConv1d


def test_from_torch():
    module = nn.Conv1d(3, 16, kernel_size=5)

    enc = EncryptedConv1d.from_torch(module)

    assert enc.in_channels == 3
    assert enc.out_channels == 16
    assert enc.kernel_size == 5
    assert enc.stride == 1
    assert enc.padding == 0
    assert enc.groups == 1
    assert enc.dilation == 1
    assert enc.weight.dtype == torch.float64
    assert enc.weight.device.type == "cpu"
    assert enc.weight_matrix.shape == (16, 15)


def test_weight_matrix_shape():
    weight = torch.randn(7, 3, 5)

    enc = EncryptedConv1d(3, 7, kernel_size=5, weight=weight)

    assert enc.weight_matrix.shape == (7, 15)


def test_output_size_calculation():
    cases = [
        ((5, 1, 0, 1, 32), 28),
        ((3, 2, 1, 1, 17), 9),
        ((3, 1, 0, 2, 10), 6),
        ((4, 3, 2, 1, 15), 6),
    ]

    for (kernel_size, stride, padding, dilation, input_length), expected in cases:
        weight = torch.randn(4, 3, kernel_size)
        enc = EncryptedConv1d(
            3,
            4,
            kernel_size=kernel_size,
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        assert enc.get_output_size(input_length) == expected


def test_mult_depth():
    enc = EncryptedConv1d(3, 4, kernel_size=3, weight=torch.randn(4, 3, 3))

    assert enc.mult_depth() == 1


def test_unfold_input_1d():
    x = torch.randn(2, 3, 11)

    patches = EncryptedConv1d.unfold_input(x, kernel_size=4, stride=2, padding=1)

    expected_length = ((11 + 2 * 1 - 4) // 2) + 1
    assert patches.shape == (2, expected_length, 12)


def test_groups_support():
    module = nn.Conv1d(4, 6, kernel_size=3, groups=2, bias=False)

    enc = EncryptedConv1d.from_torch(module)

    assert enc.groups == 2
    assert enc.weight_matrix.shape == (6, 12)

    first_group_width = (module.in_channels // module.groups) * module.kernel_size[0]
    assert torch.allclose(enc.weight_matrix[0, :first_group_width], enc.weight[0].reshape(-1))
    assert torch.count_nonzero(enc.weight_matrix[0, first_group_width:]) == 0
    assert torch.count_nonzero(enc.weight_matrix[-1, :first_group_width]) == 0
    assert torch.allclose(enc.weight_matrix[-1, first_group_width:], enc.weight[-1].reshape(-1))


def test_dilation():
    module = nn.Conv1d(2, 5, kernel_size=3, dilation=2, bias=False)
    x = torch.randn(2, 13)

    enc = EncryptedConv1d.from_torch(module)
    patches = EncryptedConv1d.unfold_input(x, kernel_size=3, dilation=2)

    assert enc.dilation == 2
    assert enc.get_output_size(13) == 9
    assert patches.shape == (9, 6)
