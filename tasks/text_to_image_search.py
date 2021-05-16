# Search a set of images based on a text query
from typing import List, Tuple

from PIL import Image

import matplotlib.pyplot as plt
import torch

from text_embeddings import get_text_embeddings
from image_embeddings import get_image_embeddings


def text_to_image_search(
    search_query: str,
    list_of_images: List[Image.Image],
    image_features: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[Image.Image], List[int], List[float]]:
    """
    Search a set of image features representing images with a text query to find
    the most relevant top_k images

    Args:
        search_query: The string text to search for, such as "a cat"
        list_of_images: A list of loaded PIL.Images
        image_features: A tensor of CLIP-processed embedded image features
        top_k: The number, k, of matching images to return

    Returns:
        1. A list of PIL.Image images
        2. The indices of the k best matching images
        3. The relative probabilities of the phrase matching
    """
    embedded_query = get_text_embeddings(text_inputs=[search_query])

    similarity_scores = image_features @ embedded_query

    similarities = similarity_scores.squeeze(1)

    probabilities = 100.0 * similarity_scores

    best_matching_indices = (-similarities).argsort()

    return (
        [list_of_images[i] for i in best_matching_indices[:top_k]],
        best_matching_indices[:top_k].tolist(),
        [round(probabilities[i][0].item(), 2) for i in best_matching_indices[:top_k]],
    )


if __name__ == "__main__":
    images_to_search = [
        "../images/cat1.jpeg",
        "../images/cat2.jpeg",
        "../images/cat3.jpeg",
        "../images/cat4.png",
        "../images/dog1.jpg",
        "../images/dog2.jpeg",
        "../images/dog3.jpeg",
        "../images/house.jpg",
        "../images/lynx.jpg",
        "../images/wolf.jpg",
        "../images/sphinx.jpeg",
        "../images/cheetah.jpeg",
    ]

    # Process the images
    display_images, image_embeddings = get_image_embeddings(
        image_files=images_to_search
    )

    # Query the images
    query = "an upside-down house"

    matching_images, _, _ = text_to_image_search(
        search_query=query,
        list_of_images=display_images,
        image_features=image_embeddings,
    )

    plt.figure(figsize=(4, 4))
    plt.title(f"Searching for {query}")
    for i, image in enumerate(matching_images):
        plt.subplot(1, len(matching_images), i + 1)
        plt.imshow(image)

        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig("images/text_to_image_results.png")
