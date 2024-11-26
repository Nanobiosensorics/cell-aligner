import numpy as np
import unittest

from alignment import find_translation_pmc, find_translation_stochastic


class MethodsTest(unittest.TestCase):
  def test_find_translation_stochastic(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = find_translation_stochastic(cardio_coords, mic_coords, 1)

    # Assert
    self.assertTrue(np.allclose(-result[0], translation, 0.2))

  def test_find_translation_pmc(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = find_translation_pmc(cardio_coords, mic_coords, 1)

    # Assert
    self.assertTrue(np.allclose(-result[0], translation, 0.2))