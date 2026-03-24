import torch
import numpy as np

def activations_view(layer: int,  activations: torch.Tensor) -> float:
    print(f"in activations_view")
    print(f"---- layer ----", layer)
    print(activations.shape)
    print(sum(activations))
    print(activations)
