import matplotlib
import glob
import os
import cv2
import click
import pickle
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


@click.command()
@click.option("--input_path", type=str, required=True, help="A valid path to read microscope and biosensor data from.")
@click.option("--output_path", type=str, required=True, help="A valid path to to store results to.")
@click.option("--cellpose_model_path", type=str, required=True, help="A valid path to a cellpose model to use for segmentation.")
@click.option("--flip", type=str, required=True, help="1 if flip on axis, 0 if not. (e.g. 1,0)")
def process(input_path, output_path, cellpose_model_path, flip):
  flip = __parse_1d_int_array(flip)

  well_data = __read_biosensor_data(os.path.join(input_path, "epic_data"), flip)
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
def __read_biosensor_data(input_path: str, flip: list[bool]):
  preprocessing_params["flip"] = flip

  raw_wells, full_time, full_phases = load_data(input_path, flip=preprocessing_params["flip"])
  _, _, _, filter_ptss, selected_range = preprocessing(preprocessing_params, raw_wells, full_time, full_phases, background_coords=filter_params)
  localized_well_data = localization(preprocessing_params, localization_params, raw_wells, selected_range, filter_ptss)

  return localized_well_data


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


# Parses a command line string input as an int array.
def __parse_1d_int_array(str):
  array_1d = list(map(int, str.split(',')))
  return array_1d


if __name__ == '__main__':
    process()