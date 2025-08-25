"""
Color Clustering with K‑Means
----------------------------

This script demonstrates unsupervised learning with K‑Means clustering
applied to image pixel data.  The goal is to identify the most
dominant colours in an image by grouping similar RGB values together.
Such a technique can be used for palette extraction, image
compression, or artistic effects.

Features:

* Reads an input image from disk, or generates a random test image if
  none is provided.
* Reshapes the image into a 2D array of pixels and applies K‑Means
  clustering.
* Computes the proportion of pixels assigned to each cluster and
  displays the cluster centres as RGB values.
* Generates a palette image that visualises the dominant colours.

Usage:
    python color_clustering.py --input path/to/image.png --clusters 5

If no `--input` argument is given, a random sample image will be
created.  The palette will be saved as `palette.png` in the current
directory.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


def generate_random_image(width: int = 300, height: int = 300, blocks: int = 5) -> Image.Image:
    """Generate a synthetic image composed of coloured rectangles.

    Parameters
    ----------
    width : int
        Width of the generated image.
    height : int
        Height of the generated image.
    blocks : int
        Number of coloured blocks per row/column.

    Returns
    -------
    Image.Image
        A PIL image with random colours.
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    block_w = width // blocks
    block_h = height // blocks
    for i in range(blocks):
        for j in range(blocks):
            color = tuple(np.random.randint(0, 256, size=3))
            draw.rectangle([
                (i * block_w, j * block_h),
                ((i + 1) * block_w - 1, (j + 1) * block_h - 1)
            ], fill=color)
    return img


def extract_palette(image: Image.Image, n_clusters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Apply K‑Means clustering to extract dominant colours from an image.

    Parameters
    ----------
    image : Image.Image
        Input image.
    n_clusters : int
        Number of colour clusters to find.

    Returns
    -------
    cluster_centres : np.ndarray of shape (n_clusters, 3)
        RGB values of cluster centres in the range [0, 255].
    proportions : np.ndarray of shape (n_clusters,)
        Fraction of total pixels assigned to each cluster.
    """
    # Convert image to numpy array of shape (num_pixels, 3)
    data = np.array(image)
    pixels = data.reshape((-1, 3)).astype(float)
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centres = kmeans.cluster_centers_
    # Compute proportions
    counts = np.bincount(labels, minlength=n_clusters)
    proportions = counts / counts.sum()
    return centres, proportions


def save_palette(centres: np.ndarray, proportions: np.ndarray, filename: str = "palette.png", width: int = 300, height: int = 50) -> None:
    """Save a visual palette representing the dominant colours.

    Parameters
    ----------
    centres : np.ndarray
        RGB cluster centres.
    proportions : np.ndarray
        Corresponding proportions of each colour.
    filename : str
        Name of the output image file.
    width : int
        Width of the palette image.
    height : int
        Height of the palette image.
    """
    palette = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(palette)
    start = 0
    for colour, prop in zip(centres, proportions):
        end = start + prop * width
        draw.rectangle([(start, 0), (end, height)], fill=tuple(map(int, colour)))
        start = end
    palette.save(filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dominant colours from an image using K‑Means clustering.")
    parser.add_argument("--input", type=str, default=None, help="Path to the input image (PNG/JPEG). If omitted, a random image is generated.")
    parser.add_argument("--clusters", type=int, default=5, help="Number of colour clusters to identify.")
    args = parser.parse_args()

    if args.input and os.path.isfile(args.input):
        img = Image.open(args.input).convert("RGB")
    else:
        print("No valid input image provided; generating a random test image.")
        img = generate_random_image()
        img.save("sample_image.png")
        print("Sample image saved as sample_image.png")
    centres, proportions = extract_palette(img, args.clusters)
    print("Dominant colours (RGB) and their proportions:")
    for idx, (centre, prop) in enumerate(zip(centres, proportions)):
        rgb = tuple(map(int, centre))
        print(f"Cluster {idx + 1}: {rgb}, proportion: {prop:.2f}")
    save_palette(centres, proportions, filename="palette.png")
    print("Palette image saved as palette.png")


if __name__ == "__main__":
    main()
