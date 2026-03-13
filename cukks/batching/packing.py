"""
Slot packing utilities for batched CKKS inference.

This module provides utilities for packing multiple samples into a single
ciphertext, enabling efficient SIMD-style batch processing.
"""

from __future__ import annotations

from typing import List

import torch


class SlotPacker:
    """Packs multiple samples into CKKS slots for batch processing.
    
    CKKS ciphertexts have a fixed number of "slots" (typically N/2 where N is the
    ring dimension). This class efficiently packs multiple samples into these slots,
    enabling single-instruction-multiple-data (SIMD) style batch processing.
    
    Example:
        >>> packer = SlotPacker(slots_per_sample=10, total_slots=8192)
        >>> samples = [torch.randn(10) for _ in range(4)]
        >>> packed = packer.pack(samples)  # Shape: (40,) packed contiguously
        >>> recovered = packer.unpack(packed, num_samples=4)  # List of 4 tensors
    """
    
    def __init__(self, slots_per_sample: int, total_slots: int):
        """Initialize the SlotPacker.
        
        Args:
            slots_per_sample: Number of slots each sample requires.
            total_slots: Total number of slots available in the ciphertext.
                         Typically poly_mod_degree // 2.
                         
        Raises:
            ValueError: If slots_per_sample <= 0 or total_slots <= 0.
        """
        if slots_per_sample <= 0:
            raise ValueError(f"slots_per_sample must be positive, got {slots_per_sample}")
        if total_slots <= 0:
            raise ValueError(f"total_slots must be positive, got {total_slots}")
            
        self.slots_per_sample = slots_per_sample
        self.total_slots = total_slots
    
    @property
    def max_batch_size(self) -> int:
        """Maximum number of samples that can be packed."""
        return self.total_slots // self.slots_per_sample
    
    def pack(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """Pack multiple samples into a single slot vector.
        
        Samples are packed contiguously in the slot vector:
        [sample0[0], sample0[1], ..., sample1[0], sample1[1], ...]
        
        Args:
            samples: List of tensors to pack. Each tensor will be flattened.
                    All samples must have the same number of elements.
                    
        Returns:
            A 1D tensor containing all samples packed contiguously.
            
        Raises:
            ValueError: If samples list is empty.
            ValueError: If number of samples exceeds max_batch_size.
            ValueError: If any sample has more elements than slots_per_sample.
            ValueError: If samples have inconsistent sizes.
        """
        if not samples:
            raise ValueError("Cannot pack empty list of samples")
            
        num_samples = len(samples)
        if num_samples > self.max_batch_size:
            raise ValueError(
                f"Number of samples ({num_samples}) exceeds max batch size "
                f"({self.max_batch_size}). Consider using a larger poly_mod_degree "
                f"or smaller sample size."
            )
        
        flat_samples = [sample.detach().to(dtype=torch.float64).reshape(-1) for sample in samples]
        sample_sizes = [flat.numel() for flat in flat_samples]

        for i, sample_size in enumerate(sample_sizes):
            if sample_size > self.slots_per_sample:
                raise ValueError(
                    f"Sample {i} has {sample_size} elements, exceeds "
                    f"slots_per_sample={self.slots_per_sample}"
                )

        sample_size = sample_sizes[0]
        for i, current_size in enumerate(sample_sizes[1:], start=1):
            if current_size != sample_size:
                raise ValueError(
                    f"Inconsistent sample sizes: sample 0 has {sample_size} elements, "
                    f"sample {i} has {current_size} elements"
                )

        stacked = torch.stack(flat_samples, dim=0)
        if sample_size == self.slots_per_sample:
            return stacked.reshape(-1)

        packed = torch.zeros(num_samples, self.slots_per_sample, dtype=torch.float64)
        packed[:, :sample_size] = stacked
        return packed.reshape(-1)
    
    def unpack(self, packed: torch.Tensor, num_samples: int) -> List[torch.Tensor]:
        """Unpack a slot vector into individual samples.
        
        Args:
            packed: The packed slot vector.
            num_samples: Number of samples to extract.
            
        Returns:
            List of tensors, one per sample.
            
        Raises:
            ValueError: If num_samples <= 0 or exceeds max_batch_size.
            ValueError: If packed tensor is too small.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
            
        if num_samples > self.max_batch_size:
            raise ValueError(
                f"num_samples ({num_samples}) exceeds max batch size ({self.max_batch_size})"
            )
        
        required_slots = num_samples * self.slots_per_sample
        flat_packed = packed.detach().to(dtype=torch.float64).contiguous().reshape(-1)
        
        if flat_packed.numel() < required_slots:
            raise ValueError(
                f"Packed tensor has {flat_packed.numel()} elements, "
                f"but {required_slots} are required for {num_samples} samples"
            )
        
        reshaped = flat_packed[:required_slots].reshape(num_samples, self.slots_per_sample)
        return [sample.clone() for sample in reshaped.unbind(dim=0)]
    
    def __repr__(self) -> str:
        return (
            f"SlotPacker(slots_per_sample={self.slots_per_sample}, "
            f"total_slots={self.total_slots}, max_batch={self.max_batch_size})"
        )
