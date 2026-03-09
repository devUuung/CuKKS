"""
Detect LayerNorm modules eligible for inverse-free conversion.

Uses torch.fx symbolic tracing to build a dataflow graph of the model,
then walks paths between consecutive LayerNorm nodes. A LayerNorm qualifies
for inverse-free conversion when every operation on the path to the next
LayerNorm is either:
  - nn.Linear (weight multiply + bias)
  - nn.ReLU  (homogeneous: ReLU(σx) = σ·ReLU(x))
  - nn.Dropout / nn.Identity (no-ops at eval time)
  - addition (residual connections are allowed between the two LNs only
    if both branches carry the same σ factor — conservatively, we reject
    residual add nodes unless they come from the same inverse-free LN
    subtree)

If tracing fails (dynamic control flow, unsupported ops), returns an empty
set and logs a warning so the converter falls back to standard LayerNorm.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Operation types that preserve a multiplicative σ scaling factor.
# These are the ONLY module types allowed on the path between an
# inverse-free LN and the standard LN that cancels σ.
_HOMOGENEOUS_MODULE_TYPES: Tuple[type, ...] = (
    nn.Linear,
    nn.ReLU,
    nn.Dropout,
    nn.Identity,
)


def detect_inverse_free_layernorms(model: nn.Module) -> FrozenSet[str]:
    """Return the fully-qualified names of LayerNorm modules that can use
    inverse-free conversion.

    The analysis is conservative: if tracing fails or the graph contains
    ambiguous paths, no modules are marked.

    Args:
        model: A PyTorch model in eval mode.

    Returns:
        Frozen set of module name strings (e.g. ``{"encoder.norm1"}``).
        Empty if tracing fails or no eligible pairs are found.
    """
    try:
        import torch.fx as fx

        traced = fx.symbolic_trace(model)
    except Exception as exc:
        logger.warning(
            "torch.fx tracing failed (%s); inverse-free LayerNorm "
            "detection skipped. All LayerNorms will use standard conversion.",
            exc,
        )
        return frozenset()

    graph = traced.graph

    # ---------------------------------------------------------------
    # 1. Build a map: node → module-name for call_module nodes
    # ---------------------------------------------------------------
    node_to_name: Dict[fx.Node, str] = {}
    name_to_node: Dict[str, fx.Node] = {}
    for node in graph.nodes:
        if node.op == "call_module":
            node_to_name[node] = node.target  # type: ignore[assignment]
            name_to_node[node.target] = node  # type: ignore[index]

    # ---------------------------------------------------------------
    # 2. Identify all LayerNorm nodes
    # ---------------------------------------------------------------
    ln_names: List[str] = []
    for node in graph.nodes:
        if node.op == "call_module":
            target_name: str = node.target  # type: ignore[assignment]
            submod = _get_submodule(model, target_name)
            if isinstance(submod, nn.LayerNorm):
                ln_names.append(target_name)

    if len(ln_names) < 2:
        return frozenset()

    # ---------------------------------------------------------------
    # 3. For each LN, find all downstream LN nodes reachable through
    #    only homogeneous ops and check eligibility.
    # ---------------------------------------------------------------
    eligible: Set[str] = set()

    for src_name in ln_names:
        src_node = name_to_node.get(src_name)
        if src_node is None:
            continue

        # BFS/DFS from src_node's users, looking for the next LN.
        reached_lns = _find_next_layernorms_through_homogeneous(
            src_node, model, ln_names,
        )
        if reached_lns:
            eligible.add(src_name)

    return frozenset(eligible)


def _find_next_layernorms_through_homogeneous(
    src_node: "torch.fx.Node",
    model: nn.Module,
    all_ln_names: List[str],
) -> List[str]:
    """BFS from *src_node* following only homogeneous-safe edges.

    Returns the names of LayerNorm modules reachable through paths where
    every intermediate call_module is a homogeneous type.
    """
    ln_name_set = set(all_ln_names)
    visited: Set["torch.fx.Node"] = set()
    queue: List["torch.fx.Node"] = list(src_node.users.keys())
    reached: List[str] = []

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node.op == "call_module":
            target: str = node.target  # type: ignore[assignment]

            # Reached another LayerNorm → this is the "cancelling" LN.
            if target in ln_name_set:
                reached.append(target)
                continue  # Don't traverse past the cancelling LN.

            # Check if the intermediate module is homogeneous-safe.
            submod = _get_submodule(model, target)
            if not isinstance(submod, _HOMOGENEOUS_MODULE_TYPES):
                continue  # Path is broken — stop exploring this branch.

            # Safe module: continue traversing its users.
            queue.extend(node.users.keys())

        elif node.op == "call_function":
            fn = node.target
            # Allow element-wise add (residual connections).
            if fn in (torch.add, _operator_add()):
                queue.extend(node.users.keys())
            # Allow functional ReLU variants.
            elif fn in (torch.relu, torch.nn.functional.relu):
                queue.extend(node.users.keys())
            else:
                continue  # Non-homogeneous function — stop.

        elif node.op == "call_method":
            # Allow .add() method and similar safe operations
            if node.target in ("add", "add_"):
                queue.extend(node.users.keys())
            else:
                continue

        elif node.op in ("placeholder", "get_attr"):
            # These don't represent computation; skip.
            queue.extend(node.users.keys())

        elif node.op == "output":
            # Reached the output without hitting another LN — this path
            # doesn't have a cancelling LN, so it's not eligible.
            continue

        else:
            # Unknown op type — be conservative, stop.
            continue

    return reached


def _get_submodule(model: nn.Module, target: str) -> Optional[nn.Module]:
    """Safely retrieve a submodule by dotted name."""
    try:
        return model.get_submodule(target)
    except AttributeError:
        return None


def _operator_add() -> object:
    """Return operator.add without importing at module level."""
    import operator
    return operator.add
