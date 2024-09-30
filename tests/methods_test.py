import numpy as np
import unittest

import methods


class MethodsTest(unittest.TestCase):
  def test_find_translation_stochastic_speed(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = methods.find_translation_stochastic_speed(cardio_coords, mic_coords, np.array([0, 1, 2]))

    # Assert
    self.assertTrue(np.allclose(-result, translation, 0.2))

  def test_find_translation_pmc(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = methods.find_translation_pmc(cardio_coords, mic_coords, 1)

    # Assert
    self.assertTrue(np.allclose(-result, translation, 0.2))

  def test_find_translation_stochastic_memory(self):
    # Arrange
    translation = np.array([3, 2])
    mic_coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    cardio_coords = np.delete(mic_coords, [1, 2], axis=0) + translation

    # Act
    result = methods.find_translation_stochastic_memory(cardio_coords, mic_coords, np.array([0, 1, 2]))

    # Assert
    self.assertTrue(np.allclose(-result, translation, 0.2))