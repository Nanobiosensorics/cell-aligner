import numpy as np


"""
This algorithm tries to match samples of source points to each target point.
One occurence of a match is described as a translation candidate.

:param source_points: numpy array containing points which will be translated. (-1, 2) shaped.
:param target_points: numpy array containing the points where the source points will be translated to. (-1, 2) shaped.
:param source_indices: numpy array containing indices of source point samples.
"""
def find_translation_stochastic(source_points: np.ndarray, target_points: np.ndarray, source_indices: np.ndarray):
  # Select random points from source data.
  selected_source_points = source_points[source_indices]

  # Calculate all translation candidate vectors from selected source points.
  translation_candidates = (target_points - selected_source_points[:, np.newaxis]).reshape(-1, 2)
  
  # Calculate all correspondence vectors between the two datasets.
  correspondence_vectors = (target_points - source_points[:, np.newaxis]).reshape(-1, 2)

  # Calculate error vectors for each translation vector.
  error_vectors = correspondence_vectors - translation_candidates[:, np.newaxis]
  
  # Calculate the length of the error vectors.
  error_lengths = np.linalg.norm(error_vectors, axis=-1)
  
  # Calculate error scores for each candidate vector.
  error_scores = np.sum(error_lengths, axis=-1)
  
  return translation_candidates[np.argmin(error_scores)]


"""
:param source_points: numpy array containing points which will be translated. (-1, 2) shaped.
:param target_points: numpy array containing the points where the source points will be translated to. (-1, 2) shaped.
:param epsilon: pairwise consistency threshold.
"""
def find_translation_pmc(source_points: np.ndarray, target_points: np.ndarray, epsilon: float):
  pass

