import glob
import os
import cv2
import pickle
from cellpose import models
from nanobio_core.epic_cardio.processing import *
from nanobio_core.epic_cardio.processing import localization
from utils import calculate_microscope_cell_centroids
from methods import *


filter_params = {}
preprocessing_params = {
  'signal_range' : {
    'range_type': RangeType.MEASUREMENT_PHASE,
    'ranges': [0, None],
  },
  'drift_correction': {
    'threshold': 75,
    'filter_method': 'mean',
    'background_selector': False,
  },
  "flip": [False, False]
}
localization_params = {
  "threshold_range": [75, 3000],
  "neighbourhood_size": 3,
  "error_mask_filtering": True
}

model_name = "cyto3_old_annotated"
epsilon = 3


def get_localized_well_data(cardio_path):
  raw_wells, full_time, full_phases = load_data(cardio_path, flip=preprocessing_params["flip"])
  _, _, _, filter_ptss, selected_range = preprocessing(preprocessing_params, raw_wells, full_time, full_phases, background_coords=filter_params)
  localized_well_data = localization(preprocessing_params, localization_params, raw_wells, selected_range, filter_ptss)
  return localized_well_data


def get_localized_img_data(img_path):
  img_paths = glob.glob(os.path.join(img_path, "*.jpeg"))
  
  img_names, img_data = [], []
  for path in img_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    img_names.append(name)

    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_data.append(data)

  CP = models.CellposeModel(pretrained_model=f"./models/{model_name}", gpu=True)
  masks, _, _ = CP.eval(img_data, channels=[0, 0])

  img_centroids = []
  for mask in masks:
    centroids = calculate_microscope_cell_centroids(mask)
    img_centroids.append(centroids)

  result = {}
  for name, data, centroids in zip(img_names, img_data, img_centroids):
    result[name] = [data, centroids]
  
  return result


def run_eval(folder: str):
  cardio_path = os.path.join(folder, "epic_data")
  localized_well_data = get_localized_well_data(cardio_path)

  img_path = os.path.join(folder, "img_data")
  localized_img_data = get_localized_img_data(img_path)

  combined = {key: {'well_data': localized_well_data[key], 'img_data': localized_img_data[key]} for key in localized_well_data.keys()}

  for key, value in combined.items():
    stochastic_result = find_translation_stochastic(value['well_data'][1], value['img_data'][1])
    pmc_result = find_translation_pmc(value['well_data'][1], value['img_data'][1], epsilon)
    combined[key]['stochastic_result'] = stochastic_result
    combined[key]['pmc_result'] = pmc_result
  
  return combined


if __name__ == '__main__':
  result = run_eval('./data/20200722_Preo_Hela_fn')
  with open("result.pkl", "wb") as file:
    pickle.dump(result, file)

