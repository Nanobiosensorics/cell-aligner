import numpy as np

import unittest

import utils


class UtilsTest(unittest.TestCase):
  def test_calculate_microscope_cell_centroids_case_1(self):
    # Arrange
    segmentation = np.array([
      [1, 1, 2, 2, 2, 3],
      [1, 2, 2, 2, 3, 3],
      [1, 0, 0, 0, 0, 0]
    ])

    # Act
    result = utils.calculate_microscope_cell_centroids(segmentation)

    # Assert
    self.assertTrue(np.allclose(result, [[0.25, 0.75], [2.5, 0.5], [4.66666, 0.666666]]))

