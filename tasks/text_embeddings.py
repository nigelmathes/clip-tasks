# Generate embedded text labels
from pathlib import Path
from typing import List

import clip
import torch

from tqdm import tqdm

from utils import get_model_and_preprocess, get_device, load_embeddings, save_embeddings


DEFAULT_PROMPT_TEMPLATES = ["an image of {}", "a drawing of {}", "a photo of {}"]


def compute_text_embeddings(
    text_inputs: List[str], prompt_templates: List[str]
) -> torch.Tensor:
    """
    Function to perform prompt engineering and create text embeddings

    Args:
        text_inputs: Individual text inputs, such as "a cat"
        prompt_templates: Optional prepended descriptions, such as "a picture of {}"
                          NOTE: Requires "{}" to format-fill with text inputs

    Returns:
        Tensor of embedded weights
    """
    # Get torch device, CLIP model, preprocessing pipeline
    device = get_device()
    model, preprocess = get_model_and_preprocess()

    with torch.no_grad():
        text_embeddings = []
        for text_input in tqdm(text_inputs):
            # Create text prompts, combining all prompt_templates with all text_inputs
            text_prompts = [
                template.format(text_input) for template in prompt_templates
            ]
            text_prompts = clip.tokenize(text_prompts).to(device)

            # Create embeddings using CLIP model and its encoder
            encoded_text = model.encode_text(text_prompts)
            encoded_text /= encoded_text.norm(dim=-1, keepdim=True)
            text_embedding = encoded_text.mean(dim=0)
            text_embedding /= text_embedding.norm()

            # Add a given text's embeddings to the list of weights
            text_embeddings.append(text_embedding)

        # Reshape weights along axis=1
        text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    return text_embeddings


def get_text_embeddings(
    text_inputs: List[str], prompt_templates: List[str] = None
) -> torch.Tensor:
    """
    Load or compute a Tensor of text embeddings

    Args:
        text_inputs: Individual text inputs, such as "a cat"
        prompt_templates: Optional prepended descriptions, such as "a picture of {}"
                          NOTE: Requires "{}" to format-fill with text inputs

    Returns:
        Tensor of embedded weights
    """
    # Mutable default fix for a list
    if not prompt_templates:
        prompt_templates = DEFAULT_PROMPT_TEMPLATES

    try:
        text_embeddings = load_embeddings(filename=Path("outputs/text_embeddings.npy"))
    except FileNotFoundError:
        text_embeddings = compute_text_embeddings(
            text_inputs=text_inputs, prompt_templates=prompt_templates
        )
        # save_embeddings(
        #    embeddings=text_embeddings, filename=Path("outputs/text_embeddings.npy")
        # )

    return text_embeddings
