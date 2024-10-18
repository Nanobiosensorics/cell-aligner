import numpy as np


"""
This algorithm tries to match samples of source points to each target point.
One occurence of a match is described as a translation candidate.
This method uses the combination of standard python operations and numpy operations to save memory.

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


"""
:param source_points: numpy array containing points which will be translated. (-1, 2) shaped.
:param target_points: numpy array containing the points where the source points will be translated to. (-1, 2) shaped.
:param epsilon: pairwise consistency threshold.
"""
def find_translation_pmc(source_points: np.ndarray, target_points: np.ndarray, epsilon: float):
  def is_pairwise_consistent(corr_i: np.ndarray, corr_j: np.ndarray):
    return np.linalg.norm(corr_i - corr_j) <= epsilon
  
  def get_subgraph_adjacencies(nodes: np.ndarray):
    return [np.intersect1d(corr_adj[nodes[i]], nodes) for i in range(len(nodes))]
  
  def get_minimal_required_candidates(candidates: np.ndarray, removed: np.ndarray):
    # We need to choose nodes from the current branch.
    all = np.concatenate((candidates, removed))
    
    # We want the adjacency which stops most nodes from expanding.
    best = candidates
    for node in all:
      diff = np.setdiff1d(candidates, corr_adj[node])
      if len(diff) < len(best):
        best = diff
    
    return best
  
  def find_coloring_greedy(candidates):
    def find_mex(colors: np.ndarray):
      # Mex can be found in linear time if the array is sorted.
      colors = np.sort(colors)

      # We can take 0, because all other numbers are greater.
      if colors[0] != 0:
        return 0
      
      # If we have a gap between two consecutive numbers we can take that.
      for i in range(1, len(colors)):
        if colors[i-1] + 1 < colors[i]:
          return colors[i-1] + 1
      
      return colors[-1] + 1

    n_candidates = len(candidates)

    subgraph_adj = get_subgraph_adjacencies(candidates)
    
    # Mapping between graph node id and candidate id.
    index = {}
    for i in range(n_candidates):
      index[candidates[i]] = i
    
    # Initializing coloring.
    candidates_coloring = np.zeros(n_candidates)

    for i in range(n_candidates):
      # This guarantees that mex will be greater than 0.
      curr_colors = [0]
      # Getting all the colors of the neighbours.
      for neighbour in subgraph_adj[i]:
        curr_colors.append(candidates_coloring[index[neighbour]])
      curr_colors = np.array(curr_colors)
      # The current node's color will be the mex of all the colors.
      candidates_coloring[i] = find_mex(curr_colors)
    
    global_coloring = {}
    for i in range(n_candidates):
      global_coloring[candidates[i]] = candidates_coloring[i]
  
    return global_coloring
  
  # Calculate all correspondence vectors between the two datasets and get its length.
  corr_vectors = (target_points - source_points[:, np.newaxis]).reshape(-1, 2)
  n_corrs = len(corr_vectors)

  # Initializing adjacency list to represent the graphs.
  corr_adj = [[] for _ in range(n_corrs)]

  # Building the graph by adding the adjacencies' of each correspondence.
  for i in range(n_corrs):
    for j in range(i+1, n_corrs):
      if is_pairwise_consistent(corr_vectors[i], corr_vectors[j]):
        corr_adj[i].append(j)
        corr_adj[j].append(i)
  
  # Converting each adjacency list to sorted numpy arrays.
  for i in range(n_corrs):
    corr_adj[i] = np.sort(np.array(corr_adj[i]))  

  best_clique = np.array([])
  curr_clique = np.array([])

  # Reordering in decreasing order of degrees to minimize branching factor.
  initial_candidates = np.arange(0, n_corrs, dtype=np.int32)
  initial_candidates = initial_candidates[np.argsort([len(adj) for adj in corr_adj])]

  initial_removed = np.array([], dtype=np.int32)

  def evaluate_branch(candidates: np.ndarray, removed: np.ndarray):
    nonlocal best_clique
    nonlocal curr_clique

    coloring = find_coloring_greedy(candidates)
    def get_colors_of_candidates(candidates):
      return np.array([coloring[node] for node in candidates])

    candidates = candidates[np.argsort(get_colors_of_candidates(candidates))]
    required_candidates = get_minimal_required_candidates(candidates, removed)

    i = len(candidates) - 1

    while i >= 0:
      if len(curr_clique) + np.max(get_colors_of_candidates(candidates)) <= len(best_clique):
        return
      
      vi = candidates[i]
      if vi in required_candidates:
        curr_clique = np.concatenate((curr_clique, [vi]))
        next_candidates = np.intersect1d(candidates, corr_adj[vi])

        if len(next_candidates) > 0:
          next_removed = np.intersect1d(removed, corr_adj[vi])
          evaluate_branch(next_candidates, next_removed)
        elif len(curr_clique) > len(best_clique):
          best_clique = curr_clique
        
        curr_clique = np.setdiff1d(curr_clique, [vi])
        candidates = np.setdiff1d(candidates, [vi])
        removed = np.concatenate((removed, [vi]))

        if coloring[vi] not in get_colors_of_candidates(candidates):
          for c in candidates:
            if coloring[c] > coloring[vi]:
              coloring[c] -= 1
      i -= 1
  
  evaluate_branch(initial_candidates, initial_removed)
  return np.average(corr_vectors[np.array(best_clique, dtype=np.uint32)], axis=0)