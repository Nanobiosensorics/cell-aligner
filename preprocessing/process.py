import numpy as np
import cv2


def process_microscope_data(mic_data: np.ndarray, cellpose_model):
  from utils import calculate_microscope_cell_centroids

  mask, _, _ = cellpose_model.eval(mic_data, channels=[0, 0])
  centroids = calculate_microscope_cell_centroids(mask)
  return (mic_data, centroids)


def process_biosensor_data(well_data: np.ndarray, params: dict):
  from nanobio_core.epic_cardio.data_correction import correct_well
  from nanopyx.methods import SRRF
  from nanobio_core.epic_cardio.math_ops import calculate_cell_maximas
  from nanobio_core.image_fitting.cardio_mic import CardioMicFitter, CardioMicScaling

  slicer = slice(int(params['preprocessing_params']['range_lowerbound'] * len(well_data)), len(well_data))

  # Extracting raw well data.
  raw_well = well_data
  if len(raw_well.shape) > 2:
    raw_well = np.max(raw_well, axis=0)

  # Correcting well
  well_data, _, _ = correct_well(
    well_data[slicer], coords=[], 
    threshold=params['preprocessing_params']['drift_correction']['threshold'], 
    mode=params['preprocessing_params']['drift_correction']['filter_method'])
  
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
  processed_well = well_data
  if len(processed_well.shape) > 2:
    processed_well = np.max(processed_well, axis=0)
  processed_well = cv2.resize(processed_well, (size, size), interpolation=cv2.INTER_NEAREST)
  ptss = ptss * size / 80 / magnification

  return (processed_well, ptss, raw_well)

