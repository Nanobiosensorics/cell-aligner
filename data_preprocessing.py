import glob
import os
import cv2
import pickle
import numpy as np
from nanobio_core.epic_cardio.processing import load_data
from nanobio_core.epic_cardio.data_correction import correct_well
from nanobio_core.epic_cardio.math_ops import calculate_cell_maximas
from nanobio_core.image_fitting.cardio_mic import CardioMicFitter, CardioMicScaling
from nanopyx.methods import SRRF
from cellpose import models
from utils import calculate_microscope_cell_centroids


def process(input_path: str, output_path: str, cellpose_model_path: str, scaling: str, magnification: int, params: dict):
  well_data = __read_biosensor_data(os.path.join(input_path, "epic_data"), scaling, magnification, params)
  microscope_data = __read_microscope_data(os.path.join(input_path, "img_data"), cellpose_model_path)

  result = {}
  for key in well_data.keys():
    result[key] = {
      "well_data": well_data[key],
      "microscope_data": microscope_data[key]
    }
  
  with open(output_path, "wb") as file:
    pickle.dump(result, file)


# Reads and preprocesses biosensor data.
def __read_biosensor_data(input_path: str, scaling: str, magnification: int, params: dict):
  raw_wells, _, _ = load_data(input_path, flip=params['preprocessing_params']["flip"])
  scale, _ = CardioMicFitter._get_scale(getattr(CardioMicScaling, scaling))
  result = {}

  for well_id in raw_wells.keys():
    slicer = slice(int(params['preprocessing_params']['range_lowerbound'] * len(raw_wells[well_id])), len(raw_wells[well_id]))
    well, _, _ = correct_well(raw_wells[well_id][slicer], coords=[], 
                              threshold=params['preprocessing_params']['drift_correction']['threshold'], 
                              mode=params['preprocessing_params']['drift_correction']['filter_method'])
    
    if magnification > 1:
      well = SRRF(well, magnification, 0.5)[0]
    
    ptss = calculate_cell_maximas(well,
                                  min_threshold=params['localization_params']['threshold_range'][0],
                                  max_threshold=params['localization_params']['threshold_range'][1],
                                  neighborhood_size=params['localization_params']['neighbourhood_size'],
                                  error_mask=None)
    
    max_well = well
    if len(max_well.shape) > 2:
      max_well = np.max(well, axis=0)
    max_well = cv2.resize(max_well, (scale, scale), interpolation=cv2.INTER_NEAREST)
    ptss = ptss * scale / 80 / magnification

    result[well_id] = [max_well, ptss]

  return result


# Reads and preprocesses microscope data with a specific cellpose model.
def __read_microscope_data(input_path: str, model_path: str):
  img_paths = glob.glob(os.path.join(input_path, "*.jpeg"))

  img_names, img_data = [], []
  for path in img_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    img_names.append(name)

    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_data.append(data)
  
  CP = models.CellposeModel(pretrained_model=model_path, gpu=True)
  img_centroids = []

  for name, data in zip(img_names, img_data):
    print("Parsing", name, end='\r')
    mask, _, _ = CP.eval(data, channels=[0, 0])
    centroids = calculate_microscope_cell_centroids(mask)
    img_centroids.append(centroids)
  
  result = {}
  for name, data, centroids in zip(img_names, img_data, img_centroids):
    result[name] = [data, centroids]
  
  return result