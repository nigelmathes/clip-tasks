import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from text_embeddings import get_text_embeddings
from utils import get_device, get_model_and_preprocess

device = get_device()
model, preprocess = get_model_and_preprocess()

image_files = [
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

list_of_classes = [
    "a cat",
    "a dog",
    "a pig",
    "an airplane",
    "a sphinx",
    "a wolf",
    "a lynx",
]
prompt_templates = ["an image of {}", "a drawing of {}", "a photo of {}"]

classifier_weights = get_text_embeddings(
    text_inputs=list_of_classes, prompt_templates=prompt_templates
)

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

    # logits: torch.Size([num_images, num_classes])
    logits = 100.0 * image_features.float() @ classifier_weights.float()

    # probs: torch.Size([num_images, num_classes])
    probs = logits.softmax(dim=-1)

    # probabilities & predictions: torch.Size([num_images, 1])
    probabilities, predictions = probs.topk(k=1, dim=1)

    if len(predictions) == 1:
        predicted_class = [list_of_classes[predictions]]
    else:
        predicted_class = [list_of_classes[i] for i in predictions]

print(f"Predicted classes = {predicted_class}")

plt.figure(figsize=(16, 7))

for i, image in enumerate(display_images):
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(
        f"{predicted_class[i]}, prob=" f"{round(probabilities[i][0].item() * 100, 2)}%"
    )
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("images/results.png")
# plt.show()
