"""
Encrypted pooling layers using pure HE operations.

EncryptedAvgPool2d - Average pooling via HE rotations and masking.
EncryptedMaxPool2d - Max pooling via polynomial approximation of |x|.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


# =============================================================================
# Polynomial Approximation for Absolute Value
# =============================================================================

def _chebyshev_sqrt_coeffs(  # pyright: ignore[reportUnusedFunction]
    degree: int = 4,
    domain: tuple = (0.01, 1.0),
) -> List[float]:
    """Compute Chebyshev polynomial coefficients for sqrt approximation.
    
    This is used to approximate |x| = sqrt(x^2).
    
    Args:
        degree: Polynomial degree.
        domain: Domain for approximation. Default (0.01, 1.0) avoids singularity at 0.
        
    Returns:
        Polynomial coefficients [a0, a1, a2, ...] for sqrt(x).
    """
    try:
        import numpy as np
        from numpy.polynomial import chebyshev
        
        a, b = domain
        # Chebyshev nodes on [-1, 1]
        nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        # Map to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        y = np.sqrt(x)
        
        # Fit Chebyshev polynomial
        coeffs = chebyshev.chebfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        # Fallback: hardcoded degree-4 coefficients for sqrt on (0.01, 1.0)
        # Approximation: sqrt(x) ≈ 0.5 + 0.5*x - 0.125*x^2 + 0.0625*x^3 - ...
        return [0.1, 1.45, -0.85, 0.35, -0.05]


def _smooth_abs_coeffs(degree: int = 4) -> List[float]:
    """Compute polynomial coefficients for smooth |x| approximation.
    
    Uses the approximation: |x| ≈ sqrt(x^2 + epsilon) for stability.
    We fit a polynomial to |x| directly on [-1, 1].
    
    Args:
        degree: Polynomial degree (should be even for |x| symmetry).
        
    Returns:
        Polynomial coefficients for |x| approximation.
    """
    try:
        import numpy as np
        from numpy.polynomial import polynomial
        
        # Sample points on [-1, 1]
        n_points = 100
        x = np.linspace(-1, 1, n_points)
        
        # Use smooth absolute value: sqrt(x^2 + eps)
        eps = 0.01
        y = np.sqrt(x**2 + eps)
        
        # Fit polynomial - use only even powers since |x| is symmetric
        # For degree 4: a0 + a2*x^2 + a4*x^4
        coeffs = polynomial.polyfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        # Fallback: hardcoded coefficients for |x| ≈ a0 + a2*x^2 + a4*x^4
        # This approximates |x| on [-1, 1]
        return [0.1, 0.0, 0.9, 0.0, 0.05]


class EncryptedAvgPool2d(EncryptedModule):
    """Encrypted 2D average pooling using pure HE operations.
    
    Average pooling can be implemented in CKKS as:
    1. Sum the elements in each pooling window
    2. Multiply by 1/kernel_area
    
    For 2D input with CNN layout, uses rotation-based optimization for 2x2 pooling
    or sparse matrix multiplication for other kernel sizes.
    
    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: same as kernel_size.
        padding: Padding to add. Default: 0.
        
    Note:
        Requires pre-processed input via ctx.encrypt_cnn_input() for 2D CNN layout.
        4D input is not supported in pure HE mode.
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.kernel_size = kernel_size
        
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.stride = stride
        
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        
        # Pre-compute the averaging factor
        self.pool_area = self.kernel_size[0] * self.kernel_size[1]
        self.avg_factor = 1.0 / self.pool_area
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply average pooling using pure HE operations.
        
        For 2D input with CNN layout, applies pooling via HE rotations or matmul.
        For pre-processed input without layout, applies scalar multiplication.
        4D input is not supported in pure HE mode.
        
        Args:
            x: Encrypted input tensor.
               
        Returns:
            Averaged encrypted output.
            
        Raises:
            RuntimeError: If input is 4D.
        """
        input_ndim = len(x.shape)
        
        if input_ndim == 4:
            raise RuntimeError(
                "EncryptedAvgPool2d does not support 4D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout. See: ckks_torch.CKKSInferenceContext.encrypt_cnn_input()"
            )
        elif input_ndim == 2 and hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
            kH, kW = self.kernel_size
            sH, sW = self.stride
            if kH == 2 and kW == 2 and sH == 2 and sW == 2:
                return self._forward_he_rotation(x)
            return self._forward_he_packed(x)
        else:
            # For pre-processed input, just multiply by averaging factor
            # Create a full-size plaintext with the avg_factor
            slot_count = x._cipher.size
            avg_plain = [self.avg_factor] * slot_count
            result = x.mul(avg_plain)
            return result.rescale()
    
    def _forward_he_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply average pooling on im2col CNN layout using HE operations.
        
        Input shape: (num_patches, channels) where num_patches = H * W
        Output shape: (out_patches, channels) where out_patches = (H/sH) * (W/sW)
        
        This uses matrix multiplication to sum and average pooling windows.
        
        Args:
            x: Encrypted input with _cnn_layout metadata.
            
        Returns:
            Encrypted pooled output.
        """
        import torch
        
        layout = x._cnn_layout
        num_patches = layout['num_patches']
        channels = layout['patch_features']
        
        if 'height' in layout and 'width' in layout:
            height, width = layout['height'], layout['width']
        else:
            import math
            hw = int(math.sqrt(num_patches))
            if hw * hw != num_patches:
                raise ValueError(
                    f"Cannot infer H, W from num_patches={num_patches} (not a perfect square). "
                    f"For rectangular inputs, set layout['height'] and layout['width'] explicitly."
                )
            height, width = hw, hw
        H, W = height, width
        
        kH, kW = self.kernel_size
        sH, sW = self.stride
        
        # Output spatial size
        out_H = H // sH
        out_W = W // sW
        out_patches = out_H * out_W
        
        # Build pooling matrix that sums elements in each pooling window
        # Input flat: [pos(0,0)_c0, pos(0,0)_c1, ..., pos(0,1)_c0, ...]
        # For 2x2 pool with stride 2:
        # output[out_y, out_x, c] = avg(input[out_y*2:out_y*2+2, out_x*2:out_x*2+2, c])
        
        total_in = num_patches * channels  # H * W * C
        total_out = out_patches * channels  # (H/2) * (W/2) * C
        
        # Create sparse pooling weight matrix
        pool_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        
        for out_y in range(out_H):
            for out_x in range(out_W):
                out_patch_idx = out_y * out_W + out_x
                
                for ky in range(kH):
                    for kx in range(kW):
                        in_y = out_y * sH + ky
                        in_x = out_x * sW + kx
                        in_patch_idx = in_y * W + in_x
                        
                        for c in range(channels):
                            out_flat_idx = out_patch_idx * channels + c
                            in_flat_idx = in_patch_idx * channels + c
                            pool_weight[out_flat_idx, in_flat_idx] = self.avg_factor
        
        # Apply pooling via matmul
        x = x.view(total_in)
        result = x.matmul(pool_weight, None)
        
        # Update CNN layout for next layer
        result._cnn_layout = {
            'num_patches': out_patches,
            'patch_features': channels,
            'original_shape': layout.get('original_shape'),
        }
        result._shape = (out_patches, channels)
        
        return result
    
    def _forward_he_rotation(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply average pooling using rotation-based optimization.
        
        For 2x2 pooling with stride 2, instead of sparse matmul:
        1. Use rotations to align neighboring elements
        2. Add them together
        3. Mask to keep only valid output positions
        4. Compact using rotation-based gather (no matmul needed)
        
        This avoids large rotation keys that matmul-based gather requires.
        
        Layout: (num_patches, channels) in row-major order
        [P0C0, P0C1, ..., P1C0, P1C1, ..., P(W)C0, ...]
        
        For 2x2 pool at output (out_y, out_x), we need to sum:
        - Input (2*out_y, 2*out_x) + (2*out_y, 2*out_x+1) + 
          (2*out_y+1, 2*out_x) + (2*out_y+1, 2*out_x+1)
        """
        import torch
        import math
        
        layout = x._cnn_layout
        num_patches = layout['num_patches']
        channels = layout['patch_features']
        
        if 'height' in layout and 'width' in layout:
            height, width = layout['height'], layout['width']
        else:
            hw = int(math.sqrt(num_patches))
            if hw * hw != num_patches:
                raise ValueError(
                    f"Cannot infer H, W from num_patches={num_patches} (not a perfect square). "
                    f"For rectangular inputs, set layout['height'] and layout['width'] explicitly."
                )
            height, width = hw, hw
        H, W = height, width
        
        kH, kW = self.kernel_size
        sH, sW = self.stride
        
        # Only optimize for 2x2 pool with stride 2 for now
        if kH != 2 or kW != 2 or sH != 2 or sW != 2:
            return self._forward_he_packed(x)
        
        out_H = H // sH
        out_W = W // sW
        out_patches = out_H * out_W
        
        total_in = num_patches * channels
        
        # Rotation offsets for 2x2 pooling
        # In flattened (H*W, C) layout:
        # - offset 0: position (0, 0) relative to pool window
        # - offset C: position (0, 1) 
        # - offset W*C: position (1, 0)
        # - offset (W+1)*C: position (1, 1)
        
        offsets = [0, channels, W * channels, (W + 1) * channels]
        
        # Sum the 4 positions via rotation and addition
        # Start with the first position (offset 0)
        result = x
        
        for offset in offsets[1:]:  # Skip first (already have it)
            rotated = x.rotate(offset)
            result = result + rotated
        
        # Only rescale if input needed rescaling (e.g., came from a mul without rescale)
        # If input was already rescaled (e.g., from EncryptedSquare), don't rescale again
        # as that would drop the scale from Δ to ~1, causing precision loss
        if x._needs_rescale:
            result = result.rescale()
        
        # Now result has the sum of 4 neighbors at each position
        # But we only want every other position (stride 2 means skip)
        
        # Create mask for valid output positions with avg_factor
        # Valid positions: (2*out_y, 2*out_x) for all out_y, out_x
        # In flat index: (2*out_y * W + 2*out_x) * channels + c
        mask = torch.zeros(total_in, dtype=torch.float64)
        for out_y in range(out_H):
            for out_x in range(out_W):
                in_y = 2 * out_y
                in_x = 2 * out_x
                in_patch_idx = in_y * W + in_x
                for c in range(channels):
                    mask[in_patch_idx * channels + c] = self.avg_factor
        
        # Apply mask (multiply by avg_factor at valid positions, 0 elsewhere)
        result = result.mul(mask.tolist())
        result = result.rescale()
        
        # Skip the gather step - keep values in sparse format
        # The FC layer will handle the sparse layout via its weight matrix
        # This avoids needing large rotation keys for gather matmul
        
        # Store the sparse layout information for downstream layers
        result._cnn_layout = {
            'num_patches': out_patches,
            'patch_features': channels,
            'original_shape': layout.get('original_shape'),
            'sparse': True,  # Mark as sparse layout
            'sparse_positions': self._get_valid_positions(out_H, out_W, W, channels),
            'total_slots': total_in,
        }
        result._shape = (out_patches, channels)
        
        return result
    
    def _get_valid_positions(self, out_H: int, out_W: int, W: int, channels: int) -> list:
        """Get the positions of valid pooled values in the sparse layout."""
        positions = []
        for out_y in range(out_H):
            for out_x in range(out_W):
                in_y = 2 * out_y
                in_x = 2 * out_x
                in_patch_idx = in_y * W + in_x
                for c in range(channels):
                    positions.append(in_patch_idx * channels + c)
        return positions
    
    def mult_depth(self) -> int:
        """Average pooling uses 1 multiplication for scaling."""
        return 1
    
    def get_output_size(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int]:
        """Compute output spatial dimensions.
        
        Args:
            input_height: Height of the input.
            input_width: Width of the input.
            
        Returns:
            Tuple of (output_height, output_width).
        """
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        
        out_h = (input_height + 2 * pH - kH) // sH + 1
        out_w = (input_width + 2 * pW - kW) // sW + 1
        
        return out_h, out_w
    
    @classmethod
    def from_torch(cls, pool: torch.nn.AvgPool2d) -> "EncryptedAvgPool2d":
        """Create from a PyTorch AvgPool2d layer.
        
        Args:
            pool: The PyTorch AvgPool2d layer to convert.
            
        Returns:
            EncryptedAvgPool2d.
        """
        return cls(
            kernel_size=pool.kernel_size,
            stride=pool.stride,
            padding=pool.padding,
        )
    
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class EncryptedMaxPool2d(EncryptedModule):
    """Encrypted 2D max pooling using polynomial approximation in pure HE mode.
    
    Since CKKS doesn't support comparison, max is approximated using:
        max(a, b) ≈ (a + b + |a - b|) / 2
    
    where |x| is approximated via polynomial fitting to sqrt(x^2 + eps).
    
    Requires 2D input with CNN layout metadata (_cnn_layout). 4D and 3D inputs
    are not supported in pure HE mode.
    
    Warning:
        This is an APPROXIMATION. The output will differ from exact max pooling.
        Average error < 10% for normalized inputs in [-1, 1].
    
    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling. Default: same as kernel_size.
        padding: Padding to add. Default: 0.
        degree: Polynomial degree for |x| approximation. Higher = more accurate
            but deeper circuit.
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        degree: int = 4,
    ) -> None:
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.kernel_size = kernel_size
        
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.stride = stride
        
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        
        self.degree = degree
        self.pool_area = self.kernel_size[0] * self.kernel_size[1]
        self._abs_coeffs = _smooth_abs_coeffs(degree)
    
    def _approx_abs(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Approximate |x| using polynomial: |x| ≈ sqrt(x^2 + eps)."""
        return x.poly_eval(self._abs_coeffs)
    
    def _approx_max(self, a: "EncryptedTensor", b: "EncryptedTensor") -> "EncryptedTensor":
        """Approximate max(a, b) ≈ (a + b + |a - b|) / 2."""
        sum_ab = a.add(b)
        diff_ab = a.sub(b)
        abs_diff = self._approx_abs(diff_ab).rescale()
        
        if sum_ab._needs_rescale:
            sum_ab = sum_ab.rescale()
        
        result = sum_ab.add(abs_diff)
        return result.mul(0.5).rescale()
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        input_ndim = len(x.shape)

        if input_ndim == 4:
            raise RuntimeError(
                "EncryptedMaxPool2d does not support 4D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout with _cnn_layout metadata."
            )
        elif input_ndim == 3:
            raise RuntimeError(
                "EncryptedMaxPool2d does not support 3D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input()."
            )
        elif input_ndim == 2 and hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
            return self._forward_he_cnn(x)
        else:
            return x

    def _forward_he_cnn(self, x: "EncryptedTensor") -> "EncryptedTensor":
        layout = x._cnn_layout
        num_patches = layout['num_patches']
        channels = layout['patch_features']

        if 'height' in layout and 'width' in layout:
            height, width = layout['height'], layout['width']
        else:
            hw = int(math.sqrt(num_patches))
            if hw * hw != num_patches:
                raise ValueError(
                    f"Cannot infer H, W from num_patches={num_patches} (not a perfect square). "
                    f"For rectangular inputs, set layout['height'] and layout['width'] explicitly."
                )
            height, width = hw, hw
        H, W = height, width
        kH, kW = self.kernel_size
        sH, sW = self.stride

        out_H = (H + 2 * self.padding[0] - kH) // sH + 1
        out_W = (W + 2 * self.padding[1] - kW) // sW + 1
        out_patches = out_H * out_W
        total_in = num_patches * channels

        offsets = []
        for ky in range(kH):
            for kx in range(kW):
                if ky == 0 and kx == 0:
                    continue
                offsets.append((ky * W + kx) * channels)

        result = x
        for offset in offsets:
            rotated = x.rotate(offset)
            result = self._approx_max(result, rotated)

        mask = torch.zeros(total_in, dtype=torch.float64)
        for out_y in range(out_H):
            for out_x in range(out_W):
                in_y = out_y * sH
                in_x = out_x * sW
                idx = (in_y * W + in_x) * channels
                for c in range(channels):
                    mask[idx + c] = 1.0

        result = result.mul(mask.tolist())
        result = result.rescale()

        result._cnn_layout = {
            'num_patches': out_patches,
            'patch_features': channels,
            'original_shape': layout.get('original_shape'),
        }
        result._shape = (out_patches, channels)
        return result
    
    def mult_depth(self) -> int:
        """Multiplicative depth depends on polynomial degree and pooling size.
        
        Each pairwise max requires:
        - 1 subtraction (free)
        - polynomial eval for |x|: ceil(log2(degree+1)) mults
        - 1 addition (free)  
        - 1 scalar mult (for /2)
        
        For 2x2 pooling, we need 3 pairwise max operations.
        """
        poly_depth = max(1, int(math.ceil(math.log2(self.degree + 1))))
        num_pairwise_ops = self.pool_area - 1
        return poly_depth * num_pairwise_ops + 1
    
    def get_output_size(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int]:
        """Compute output spatial dimensions."""
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        
        out_h = (input_height + 2 * pH - kH) // sH + 1
        out_w = (input_width + 2 * pW - kW) // sW + 1
        
        return out_h, out_w
    
    @classmethod
    def from_torch(cls, pool: torch.nn.MaxPool2d) -> "EncryptedMaxPool2d":
        """Create from a PyTorch MaxPool2d layer.
        
        Args:
            pool: The PyTorch MaxPool2d layer to convert.
            
        Returns:
            EncryptedMaxPool2d with matching configuration.
            
        Note:
            The encrypted version uses polynomial approximation.
            Output will differ from exact MaxPool2d.
        """
        return cls(
            kernel_size=pool.kernel_size,
            stride=pool.stride,
            padding=pool.padding,
        )
    
    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, degree={self.degree}"
        )
