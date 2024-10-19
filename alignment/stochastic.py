import numpy as np


"""
This algorithm tries to match samples of source points to each target point.
One occurence of a match is described as a translation candidate.
This method uses the combination of standard python operations and numpy operations to save memory.

:param source_points: numpy array containing points which will be translated. (-1, 2) shaped.
:param target_points: numpy array containing the points where the source points will be translated to. (-1, 2) shaped.
:param source_indices_ratio: the percentage of random source samples.
"""
def find_translation_stochastic(source_points: np.ndarray, target_points: np.ndarray, source_indices_ratio: float):
  # Select random points from source data.
  np.random.seed(42)
  selected_source_points = source_points[np.random.choice(len(source_points), int(source_indices_ratio * len(source_points)), replace=False)]

  # Calculate all translation candidate vectors from selected source points.
  translation_candidates = (target_points - selected_source_points[:, np.newaxis]).reshape(-1, 2)

  # Calculate all correspondence vectors between the two datasets.
  correspondence_vectors = (target_points - source_points[:, np.newaxis]).reshape(-1, 2)

  # Storing the best average error and best translation.
  best_error, best_translation = np.Infinity, None

  # Evaluating a translation candidate by shifting all correspondence vectors and calculating
  # average length.
  def evaluate_candidate(translation_candidate: np.ndarray):
    error_vectors = correspondence_vectors - translation_candidate
    error_lengths = np.linalg.norm(error_vectors, axis=-1)
    return np.average(np.sort(error_lengths)[:len(target_points)] ** 2)

  # Tries each translation candidate and select the best.
  for candidate in translation_candidates:
    error = evaluate_candidate(candidate)
    if error < best_error:
      best_error = error
      best_translation = candidate
  
  return best_translation

