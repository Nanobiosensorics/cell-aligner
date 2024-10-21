import numpy as np
import cv2
from utils import calculate_microscope_cell_centroids
from nanobio_core.epic_cardio.data_correction import correct_well
from nanopyx.methods import SRRF
from nanobio_core.epic_cardio.math_ops import calculate_cell_maximas
from nanobio_core.image_fitting.cardio_mic import CardioMicFitter, CardioMicScaling


def process_microscope_data(mic_data: np.ndarray, cellpose_model):
  mask, _, _ = cellpose_model.eval(mic_data, channels=[0, 0])
  centroids = calculate_microscope_cell_centroids(mask)
  return (mic_data, centroids)


def process_biosensor_data(well_data: np.ndarray, params: dict):
  slicer = slice(int(params['preprocessing_params']['range_lowerbound'] * len(well_data)), len(well_data))

  # Correcting well
  well_data, _, _ = correct_well(
    well_data[slicer], coords=[], 
    threshold=params['preprocessing_params']['drift_correction']['threshold'], 
    mode=params['preprocessing_params']['drift_correction']['filter_method'])
  
  max_well = well_data
  if len(max_well.shape) > 2:
    max_well = np.max(well_data, axis=0)
  
  # Magnification if needed
  magnification = params["preprocessing_params"]["magnification"]
  if magnification > 1:
    well_data = SRRF(well_data, magnification, 0.5)[0]
  
  # Localization
  ptss = calculate_cell_maximas(
    well_data,
    min_threshold=params['localization_params']['threshold_range'][0],
    max_threshold=params['localization_params']['threshold_range'][1],
    neighborhood_size=params['localization_params']['neighbourhood_size'],
    error_mask=None)
  
  # Scaling
  size, _ = CardioMicFitter._get_scale(getattr(CardioMicScaling, params["preprocessing_params"]["scaling"]))
  max_well_scaled = cv2.resize(max_well, (size, size), interpolation=cv2.INTER_NEAREST)
  ptss = ptss * size / 80 / magnification

  return (max_well_scaled, ptss, max_well)

