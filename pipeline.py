from preprocessing import Reader, process_microscope_data, process_biosensor_data
from alignment import find_translation_stochastic, find_translation_pmc
from cellpose import models


def run_pipeline(data, mode: tuple, cellpose_model_path: str, epic_params: dict, is_processed: bool = False, only_process: bool = False):
  microscope_data, biosensor_data = {}, {}
  if isinstance(data, Reader):
    microscope_data, biosensor_data = data.read_microscope_data(), data.read_biosensor_data()
  elif isinstance(data, dict):
    for key in data.keys():
      microscope_data[key] = data[key][0]
      biosensor_data[key] = data[key][1]
  elif isinstance(data, list):
    microscope_data["_"] = data[0],
    biosensor_data["_"] = data[1]

  cellpose_model = models.CellposeModel(pretrained_model=cellpose_model_path, gpu=True)

  result = {}
  for key in microscope_data.keys():
    if is_processed:
      microscope_processed = microscope_data[key]
      biosensor_processed = biosensor_data[key]
    else:
      microscope_processed = process_microscope_data(microscope_data[key], cellpose_model)
      biosensor_processed = process_biosensor_data(biosensor_data[key], epic_params)

    if only_process:
      result[key] = (microscope_processed, biosensor_processed)
    else:
      if mode[0] == "stochastic":
        translation = -find_translation_stochastic(microscope_processed[1], biosensor_processed[1], mode[1])
      elif mode[0] == "pmc":
        translation = -find_translation_pmc(microscope_processed[1], biosensor_processed[1], mode[1])
      result[key] = (microscope_processed, biosensor_processed, translation)
  
  return result["_"] if isinstance(data, list) else result

