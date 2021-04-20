# General utilities for CLIP tasks
from typing import Tuple, Callable

import clip
import PIL
import torch


def get_device() -> str:
    """
    Set PyTorch to run on CPU or GPU based on hardware.

    Returns:
        "cuda" if a GPU is available, otherwise "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return device


def get_model_and_preprocess(
    model_name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    """
    Use clip.load() with an input model string to load the model and preprocessing
    pipeline. Defaults to the "ViT-B/32" model.

    Args:
        model_name: Name of the CLIP model, one of:
                    ["RN50", "RN101", "RN50x4", "ViT-B/32"]

    Returns:
        The loaded model and preprocessing pipeline
    """
    if model_name not in ["RN50", "RN101", "RN50x4", "ViT-B/32"]:
        raise RuntimeError(
            'Model not available. Choose one of: ["RN50", "RN101", "RN50x4", "ViT-B/32"]'
        )

    model, preprocess = clip.load(model_name, device=get_device())

    return model, preprocess
