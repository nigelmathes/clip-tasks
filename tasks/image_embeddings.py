# Generate embedded image labels
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from utils import get_model_and_preprocess, get_device, load_embeddings, save_embeddings


def compute_image_embeddings(images: List[Image.Image]) -> torch.Tensor:
    """
    Function to perform prompt engineering and create image embeddings

    Args:
        images: List of images loaded as PIL.Image.Image

    Returns:
        Tensor of embedded weights
    """
    # Get torch device, CLIP model, preprocessing pipeline
    device = get_device()
    model, preprocess = get_model_and_preprocess()

    # Process the images
    preprocessed_images = []
    for image in tqdm(images):
        # Apply CLIP preprocessing pipeline
        preprocessed_image = preprocess(image.convert("RGB"))
        preprocessed_images.append(preprocessed_image)

    image_input = torch.tensor(np.stack(preprocessed_images)).to(device)

    # Encode the images
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


def get_image_embeddings(
    image_files: List[Union[str, Path]]
) -> Tuple[List[Image.Image], torch.Tensor]:
    """
    Function to load images and compute their embeddings

    Args:
        image_files: List of paths to images to load and embed

    Returns:
        List of displayable images and a Tensor of their embedded weights
    """
    # Generate a list of loaded images in PIL.Image format to display results easily
    display_images = []
    for image_file in image_files:
        display_images.append(Image.open(image_file))

    # Load embeddings if they have already computed, otherwise compute and save them
    try:
        image_embeddings = load_embeddings(
            filename=Path("outputs/image_embeddings.npy")
        )
    except FileNotFoundError:
        image_embeddings = compute_image_embeddings(images=display_images)
        #save_embeddings(
        #    embeddings=image_embeddings, filename=Path("outputs/image_embeddings.npy")
        #)

    return display_images, image_embeddings
