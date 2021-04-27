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
) -> Tuple[List[Image.Image], List[int]]:
    """
    Search a set of image features representing images with a text query to find
    the most relevant top_k images

    Args:
        search_query:
        list_of_images:
        image_features:
        top_k:

    Returns:
        top_k relevant images based on the query, defaults to 1
    """
    embedded_query = get_text_embeddings(text_inputs=[search_query])

    similarities = (image_features @ embedded_query).squeeze(1)

    best_matching_indices = (-similarities).argsort()

    return (
        [list_of_images[i] for i in best_matching_indices[:top_k]],
        best_matching_indices[:top_k].tolist(),
    )


if __name__ == "__main__":
    images_to_search = [
        "images/cat1.jpeg",
        "images/cat2.jpeg",
        "images/cat3.jpeg",
        "images/cat4.png",
        "images/dog1.jpg",
        "images/dog2.jpeg",
        "images/dog3.jpeg",
        "images/house.jpg",
        "images/lynx.jpg",
        "images/wolf.jpg",
        "images/sphinx.jpeg",
        "images/cheetah.jpeg",
    ]

    # Process the images
    display_images, image_embeddings = get_image_embeddings(
        image_files=images_to_search
    )

    # Query the images
    query = "an upside-down house"

    matching_images, _ = text_to_image_search(
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
