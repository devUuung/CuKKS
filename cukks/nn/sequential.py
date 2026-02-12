"""
EncryptedSequential - Sequential container for encrypted modules.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Iterator, Union

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedSequential(EncryptedModule):
    """A sequential container for encrypted modules.
    
    Modules are added in order and executed sequentially.
    This mirrors torch.nn.Sequential for encrypted inference.
    
    Example:
        >>> model = EncryptedSequential(
        ...     EncryptedLinear(784, 128, w1, b1),
        ...     EncryptedReLU(),
        ...     EncryptedLinear(128, 10, w2, b2),
        ... )
        >>> output = model(encrypted_input)
        
        >>> # Or with OrderedDict
        >>> model = EncryptedSequential(OrderedDict([
        ...     ('fc1', EncryptedLinear(784, 128, w1, b1)),
        ...     ('relu', EncryptedReLU()),
        ...     ('fc2', EncryptedLinear(128, 10, w2, b2)),
        ... ]))
    """
    
    def __init__(self, *args: Union[EncryptedModule, "OrderedDict[str, EncryptedModule]"]) -> None:
        super().__init__()
        
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                if not isinstance(module, EncryptedModule):
                    raise TypeError(
                        f"Expected EncryptedModule for key '{name}', got {type(module)}"
                    )
                self.add_module(name, module)
        else:
            for idx, module in enumerate(args):
                if isinstance(module, EncryptedModule):
                    self.add_module(str(idx), module)
                else:
                    raise TypeError(f"Expected EncryptedModule, got {type(module)}")
    
    def add_module(self, name: str, module: EncryptedModule) -> None:
        """Add a module to the sequential container.
        
        Args:
            name: Name for the module.
            module: The module to add.
        """
        self.register_module(name, module)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass through all modules sequentially.
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            Encrypted output after passing through all modules.
        """
        for module in self._modules.values():
            x = x.maybe_bootstrap()
            x = module(x)
        return x
    
    def mult_depth(self) -> int:
        """Total multiplicative depth of all modules."""
        return sum(module.mult_depth() for module in self._modules.values())
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[EncryptedModule]:
        return iter(self._modules.values())
    
    def __getitem__(self, idx: Union[int, str]) -> EncryptedModule:
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self._modules)
            keys = list(self._modules.keys())
            if idx < 0 or idx >= len(keys):
                raise IndexError(f"Index {idx} out of range")
            return self._modules[keys[idx]]
        return self._modules[idx]
    
    def append(self, module: EncryptedModule) -> "EncryptedSequential":
        """Append a module to the end.
        
        Args:
            module: Module to append.
            
        Returns:
            self for chaining.
        """
        self.add_module(str(len(self._modules)), module)
        return self
