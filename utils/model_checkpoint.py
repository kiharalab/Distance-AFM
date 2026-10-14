"""Activation checkpointing for a model returning (tensor dictionary, frames)."""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_model(
    function: Callable[..., Tuple[Dict[str, torch.Tensor], torch.Tensor]],
    *args: object,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    # Reentrant checkpointing does not connect tensors nested inside dictionaries.
    # Return each tensor directly, then reconstruct the model's public output shape.
    names: Optional[Tuple[str, ...]] = None

    def tensor_outputs(*inputs: object) -> Tuple[torch.Tensor, ...]:
        nonlocal names
        outputs, frames = function(*inputs)
        current_names = tuple(outputs)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("Checkpoint recomputation changed the output keys")
        values = (frames, *outputs.values())
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("Checkpoint model outputs must all be tensors")
        return values

    tensors = checkpoint(tensor_outputs, *args)
    if names is None:
        raise RuntimeError("Checkpoint forward did not run")
    return dict(zip(names, tensors[1:])), tensors[0]
