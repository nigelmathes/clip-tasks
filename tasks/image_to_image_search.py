# Search a set of images based on a image query
from typing import List, Tuple

from PIL import Image

import torch

from image_embeddings import compute_image_embeddings


def image_to_image_search(
    image_query: Image.Image,
    list_of_images: List[Image.Image],
    image_features: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[Image.Image], List[int], List[float]]:
    """
    Search a set of image features representing images with a text query to find
    the most relevant top_k images

    Args:
        image_query: A PIL.Image image to search for in the list of images
        list_of_images: A list of loaded PIL.Images
        image_features: A tensor of CLIP-processed embedded image features
        top_k: The number, k, of matching images to return

    Returns:
        1. A list of PIL.Image images
        2. The indices of the k best matching images
        3. The relative probabilities of the phrase matching
    """
    embedded_query = compute_image_embeddings(images=[image_query])

    similarity_scores = image_features @ embedded_query

    similarities = similarity_scores.squeeze(1)

    probabilities = 100.0 * similarity_scores

    best_matching_indices = (-similarities).argsort()

    return (
        [list_of_images[i] for i in best_matching_indices[:top_k]],
        best_matching_indices[:top_k].tolist(),
        [round(probabilities[i][0].item(), 2) for i in best_matching_indices[:top_k]],
    )
