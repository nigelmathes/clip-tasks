# Generate embedded image labels
from typing import List, Tuple, Union
from pathlib import Path

import clip
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from utils import get_model_and_preprocess, get_device


def image_embeddings(image_files: List[Union[str, Path]]) -> Tuple[List[torch.Tensor],
                                                    List[Image.Image]]:
    """
    Placeholder

    Returns:

    """
    # Get torch device, CLIP model, preprocessing pipeline
    device = get_device()
    model, preprocess = get_model_and_preprocess()

    # Process the images
    images = []
    display_images = []
    for image_file in image_files:
        display_image = Image.open(image_file)
        image = preprocess(display_image.convert("RGB")).to(device)
        images.append(image)
        display_images.append(display_image)

    image_input = torch.tensor(np.stack(images)).to(device)

    # predict
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features, display_images
