## Colour Clustering with K Means

This project explores **unsupervised learning** by applying K‑Means
clustering to the pixel values of an image.  The goal is to extract the
most prominent colours in the image and display them as a palette.  This
technique is widely used in computer vision for tasks such as colour
quantisation, palette extraction and image compression.

### What it does

* Reads an input image (PNG/JPEG) or generates a synthetic test image
  composed of coloured rectangles.
* Treats each pixel’s RGB value as a data point in 3‑D colour space.
* Runs the K‑Means algorithm to partition the pixels into `k` clusters
  based on colour similarity.
* Computes the proportion of pixels belonging to each cluster.
* Saves a palette image where each colour’s width is proportional to
  its frequency in the original image.

### Why it matters

K‑Means clustering is a fundamental unsupervised learning algorithm used
for pattern discovery and dimensionality reduction.  Applying it to
pixel data demonstrates how unsupervised techniques can summarise
complex data into a few representative components.  Websites like
MachineLearningProjects.net list projects such as “finding the most
dominant colours in an image” among recommended machine learning
projects【818819262742885†L175-L181】 because they illustrate core concepts and
are visually appealing.

### Running the project

1. Install the required dependencies (`pillow` and `scikit‑learn`).
2. Open a terminal in this directory.
3. To analyse a custom image, run:

   ```bash
   python color_clustering.py --input path/to/your/image.jpg --clusters 5
   ```

   Replace `path/to/your/image.jpg` with the path to your own image and
   adjust `--clusters` to change the number of colours in the palette.

4. If you omit the `--input` argument, the script generates a random
   test image and saves it as `sample_image.png`.  The palette is saved
   as `palette.png`.

### Extensions

* **Image compression** – Replace all pixels in the original image with
  their cluster centre to produce a posterised version with fewer
  colours.
* **Interactive tool** – Create a simple web interface where users
  upload an image and receive the extracted palette.
* **Comparison of algorithms** – Experiment with other clustering
  algorithms such as DBSCAN or Mean Shift to see how they perform on
  colour data.
