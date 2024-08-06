import numpy as np


"""
:param segmentation: microscope image segmentation mask where each cell instance
is marked with an unique number started from 1. (cell ids are 1, 2, 3...)
:return: an array where the ith indexed centroid corresponds to the (i+1)th cell id.
"""
def calculate_microscope_cell_centroids(segmentation: np.ndarray):
  # Calculating the number of cells on the image.
  mx_id = np.max(segmentation)

  # For every cell instance we have a centroid.
  result = np.zeros((mx_id, 2))

  # Calculating cell centroids for each cell id.
  for id in range(1, mx_id + 1):
    # Getting x and y indices of the current cell.
    indices = np.where(segmentation == id)

    # Getting coords by averaging cell indices.
    y_mean = np.mean(indices[0])
    x_mean = np.mean(indices[1])

    result[id - 1] = np.array([x_mean, y_mean])

  return result

