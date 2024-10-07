import click
import os
import numpy as np
import pickle
from data_reading import read_biosensor_data, read_microscope_data
from methods import find_translation_stochastic


@click.group()
def cli():
  """Main entry point for the CLI."""
  pass


@cli.command()
@click.option("--input_path", type=str, required=True, help="A valid path to read microscope and biosensor data from.")
@click.option("--output_path", type=str, required=True, help="A valid path to to store results to.")
@click.option("--cellpose_model_path", type=str, required=True, help="A valid path to a cellpose model to use for segmentation.")
@click.option("--batch_size", type=str, required=True, help="Batch size to evaluate cellpose model on. (important for memory)")
@click.option("--flip", type=str, required=True, help="1 if flip on axis, 0 if not. (e.g. 1,0)")
@click.option("--source_indices_ratio", type=str, required=True, help="Ratio which determines how many source indices we are taking.")
def eval_stochastic(input_path, output_path, cellpose_model_path, batch_size, flip, source_indices_ratio):
  batch_size = int(batch_size)
  flip = __parse_1d_int_array(flip)
  source_indices_ratio = float(source_indices_ratio)

  flip[0] = False if flip[0] == 0 else True
  flip[1] = False if flip[1] == 0 else True

  well_data = read_biosensor_data(os.path.join(input_path, "epic_data"), flip)
  microscope_data = read_microscope_data(os.path.join(input_path, "img_data"), cellpose_model_path, batch_size)

  np.random.seed(42)
  result = {}
  for key in well_data.keys():
    curr_well = well_data[key]
    curr_microscope = microscope_data[key]

    num_indices = len(curr_microscope[1])
    random_indices = np.random.choice(num_indices, int(source_indices_ratio * num_indices), replace=False)
    translation = find_translation_stochastic(curr_microscope[1], curr_well[1], random_indices)

    result[key] = {
      "well_data": curr_well,
      "microscope_data": curr_microscope,
      "translation": translation
    }
  
  with open(output_path, "wb") as file:
    pickle.dump(result, file)


# Parses a command line string input as an int array.
def __parse_1d_int_array(str):
  array_1d = list(map(int, str.split(',')))
  return array_1d


if __name__ == "__main__":
  cli()