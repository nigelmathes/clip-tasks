# CLIP Tasks
A set of tasks CLIP can help perform without training, combining text and imagery.

A mashup of a few CLIP repos and my own code, based on:
- https://github.com/woctezuma/heroku-clip
- https://github.com/haofanwang/natural-language-joint-query-search

# Install
Clone the repo! Then,
```bash
pip install -r requirements.txt
```

# Image Labeling
First, make an `outputs` directory under `clip-tasks`. Then, In `tasks/infer_image_label.
py`, edit the list of images and list of text queries, and then run it. 

Plots should be saved in the `outputs` directory.

# Text to Image Search
First, make an `images` directory under `clip-tasks`. Then, In 
`tasks/text_to_image_search. py`, edit the list of images and list of text queries,
and then run it. 

The matched image should be saved in the `images` directory.

# Image to Image Search
Soon!

# Image + Text to Image Search
Soon!

# Text Video Search
```bash
streamlit run tasks/video_search_app.py --server.port 80
```

# Image to Video Search
Soon!

# Image + Text to Video Search
Soon!