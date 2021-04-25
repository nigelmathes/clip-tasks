import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

import cv2
import youtube_dl

from PIL import Image

from image_embeddings import compute_image_embeddings
from text_to_image_search import text_to_image_search

video_to_download = "https://youtu.be/-ssXJtzFOjA"

with TemporaryDirectory() as download_directory:
    # Download the video as 'video.mp4' in the temporary directory
    # See: https://stackoverflow.com/a/63002071 for how to get more info
    youtube_dl_options = {
        "outtmpl": f"{download_directory}/video.mp4",
        "extractaudio": False
    }

    # Perform the download
    with youtube_dl.YoutubeDL(youtube_dl_options) as ydl:
        ydl.download([video_to_download])

    # Iterate through video, extracting a frame every second
    path_to_video = f'{download_directory}/video.mp4'
    video = cv2.VideoCapture(path_to_video)

    # Get frames per second to load a frame every second
    fps = video.get(cv2.CAP_PROP_FPS)
    multiplier = int(fps * 2)

    images = []

    while True:
        success, frame = video.read()

        if not success:
            break

        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_number % multiplier == 0:
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    video.release()

image_embeddings = compute_image_embeddings(images=images)

text_query = 'two monkeys'

matching_images = text_to_image_search(
    search_query=text_query,
    list_of_images=images,
    image_features=image_embeddings,
    top_k=5
)

plt.figure(figsize=(8, 4))
plt.title(f"Searching for {text_query}")
for i, image in enumerate(matching_images):
    plt.subplot(1, len(matching_images), i + 1)
    plt.imshow(image)

    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("images/video_search_results.png")

print("Done!")