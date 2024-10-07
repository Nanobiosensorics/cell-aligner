import matplotlib
import glob
import os
import cv2
from nanobio_core.epic_cardio.processing import load_data, preprocessing, localization, RangeType
from cellpose import models
from utils import calculate_microscope_cell_centroids


# Need to set this to work.
matplotlib.use("agg")


# Processing parameters.
preprocessing_params = {
  'signal_range' : {
    'range_type': RangeType.MEASUREMENT_PHASE,
    'ranges': [0, None],
  },
  'drift_correction': {
    'threshold': 75,
    'filter_method': 'mean',
    'background_selector': False,
  }
}
localization_params = {
  "threshold_range": [75, 3000],
  "neighbourhood_size": 3,
  "error_mask_filtering": True
}
filter_params = {}


def read_biosensor_data(input_path: str, flip: list[bool]):
  preprocessing_params["flip"] = flip

  raw_wells, full_time, full_phases = load_data(input_path, flip=preprocessing_params["flip"])
  _, _, _, filter_ptss, selected_range = preprocessing(preprocessing_params, raw_wells, full_time, full_phases, background_coords=filter_params)
  localized_well_data = localization(preprocessing_params, localization_params, raw_wells, selected_range, filter_ptss)

  return localized_well_data


def read_microscope_data(input_path: str, model_path: str, batch_size: int):
  img_paths = glob.glob(os.path.join(input_path, "*.jpeg"))

  img_names, img_data = [], []
  for path in img_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    img_names.append(name)

    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_data.append(data)
  
  CP = models.CellposeModel(pretrained_model=model_path, gpu=True)

  masks, _, _ = CP.eval(img_data, channels=[0, 0], batch_size=batch_size)

  img_centroids = []
  for mask in masks:
    centroids = calculate_microscope_cell_centroids(mask)
    img_centroids.append(centroids)
  
  result = {}
  for name, data, centroids in zip(img_names, img_data, img_centroids):
    result[name] = [data, centroids]
  
  return result

