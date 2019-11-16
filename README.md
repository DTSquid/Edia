# Edia
Python Functions and Classes for image manipulation and analysis.
## Prerequisites
```
numpy
cv2
sklearn.cluster
PIL
```
### Getting Started
first open an image using PIL
```
from PIL import Image

path = 'C:\\path\\to\\image.png'
img = Image.open(path).convert('RGB')
```
### Break image into clusters
Using a mean shift algorithm
```
clusters = get_ms_clusters(img)
```
Using a K-means algorithm
```
clusters = get_kmeans_clusters(img)
```
These functions return a numpy array with the same dimensions as the image input.  The clusters are represented by integers in this numpy array.  
The clustered image can be visualized using pyplot
```
import matplotlib.pyplot as plt
plt.imshow(clusters)
```
Get the mean color in each cluster
```
palette = get_cluster_colors(clusters,img)
```
Get an image colored using the mean color in each cluster
```
colored_Image = color_clusters_image(clusters,img)
```
