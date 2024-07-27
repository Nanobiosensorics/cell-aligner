import numpy as np
import unittest

import methods


class MethodsTest(unittest.TestCase):
  def test_find_translation_stochastic(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = methods.find_translation_stochastic(cardio_coords, mic_coords, np.array([0, 1, 2]))

    # Assert
    self.assertTrue(np.allclose(-result, translation))

  def test_find_translation_pmc(self):
    # Arange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = methods.find_translation_pmc(cardio_coords, mic_coords, 1)

    # Assert
    self.assertTrue(np.allclose(-result, translation, 0.5))


if __name__ == '__main__':
  unittest.main()