"""
EncryptedModule - Base class for encrypted neural network modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedModule(ABC):
    """Base class for all encrypted neural network modules.
    
    This mirrors torch.nn.Module but is designed specifically for
    inference on encrypted data. Unlike torch.nn.Module:
    
    - No gradient tracking (inference only)
    - Parameters are stored as plaintext (weights are not encrypted)
    - Input/output are EncryptedTensor objects
    
    Example:
        >>> class MyEncryptedLayer(EncryptedModule):
        ...     def __init__(self, weight):
        ...         super().__init__()
        ...         self.weight = weight
        ...     
        ...     def forward(self, x):
        ...         return x.matmul(self.weight)
    """
    
    def __init__(self) -> None:
        self._modules: Dict[str, "EncryptedModule"] = {}
        self._parameters: Dict[str, Any] = {}
        self._name: Optional[str] = None
    
    @abstractmethod
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass on encrypted input.
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            Encrypted output tensor.
        """
        pass
    
    def __call__(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Make the module callable."""
        return self.forward(x)
    
    def register_module(self, name: str, module: Optional["EncryptedModule"]) -> None:
        """Register a child module.
        
        Args:
            name: Name of the child module.
            module: The module to register.
        """
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
    
    def register_parameter(self, name: str, param: Any) -> None:
        """Register a parameter (weight/bias).
        
        Args:
            name: Name of the parameter.
            param: The parameter value (typically a torch.Tensor).
        """
        self._parameters[name] = param
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, EncryptedModule):
            self._modules[name] = value
        elif hasattr(self, '_modules') and name in self._modules:
            del self._modules[name]
        object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str) -> Any:
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_parameters' in self.__dict__:
            parameters = self.__dict__['_parameters']
            if name in parameters:
                return parameters[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def modules(self) -> Iterator["EncryptedModule"]:
        """Iterate over all modules (including self).
        
        Yields:
            Each module in the tree.
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "EncryptedModule"]]:
        """Iterate over all modules with their names.
        
        Args:
            prefix: Prefix for module names.
            
        Yields:
            Tuples of (name, module).
        """
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(submodule_prefix)
    
    def children(self) -> Iterator["EncryptedModule"]:
        """Iterate over immediate children modules.
        
        Yields:
            Each child module.
        """
        yield from self._modules.values()
    
    def named_children(self) -> Iterator[Tuple[str, "EncryptedModule"]]:
        """Iterate over immediate children with their names.
        
        Yields:
            Tuples of (name, module).
        """
        yield from self._modules.items()
    
    def parameters(self) -> Iterator[Any]:
        """Iterate over all parameters.
        
        Yields:
            Each parameter in the module tree.
        """
        yield from self._parameters.values()
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        """Iterate over all parameters with their names.
        
        Args:
            prefix: Prefix for parameter names.
            
        Yields:
            Tuples of (name, parameter).
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for mod_name, module in self._modules.items():
            submodule_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(submodule_prefix)
    
    def mult_depth(self) -> int:
        """Estimate the multiplicative depth of this module.
        
        Returns:
            Estimated number of multiplicative operations.
        """
        return 0  # Override in subclasses
    
    def extra_repr(self) -> str:
        """Extra information for repr.
        
        Override this to add layer-specific information.
        """
        return ""
    
    def __repr__(self) -> str:
        extra = self.extra_repr()
        if self._modules:
            child_lines = []
            for name, module in self._modules.items():
                mod_str = repr(module).replace('\n', '\n  ')
                child_lines.append(f"  ({name}): {mod_str}")
            children_str = '\n'.join(child_lines)
            if extra:
                return f"{self.__class__.__name__}({extra},\n{children_str}\n)"
            return f"{self.__class__.__name__}(\n{children_str}\n)"
        if extra:
            return f"{self.__class__.__name__}({extra})"
        return f"{self.__class__.__name__}()"


class EncryptedIdentity(EncryptedModule):
    """Identity layer that passes input through unchanged.
    
    This is used as a no-op placeholder, e.g., for Dropout layers
    which have no effect during inference.
    """
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Pass input through unchanged."""
        return x
    
    def mult_depth(self) -> int:
        """Identity has zero multiplicative depth."""
        return 0
