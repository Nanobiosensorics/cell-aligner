import click
import numpy as np
import pickle
from methods import find_translation_stochastic, find_translation_pmc


@click.group()
def cli():
  """Main entry point for the CLI."""
  pass


@cli.command()
@click.option("--input_path", type=str, required=True, help="A valid path to read preprocessed microscope and biosensor data from.")
@click.option("--output_path", type=str, required=True, help="A valid path to to store results to.")
@click.option("--source_indices_ratio", type=str, required=True, help="Ratio which determines how many source indices we are taking.")
def eval_stochastic(input_path, output_path, source_indices_ratio):
  source_indices_ratio = float(source_indices_ratio)

  with open(input_path, "rb") as file:
    data = pickle.load(file)

  np.random.seed(42)
  result = {}
  for key in data.keys():
    curr_well = data[key]["well_data"]
    curr_microscope = data[key]["microscope_data"]

    num_indices = len(curr_microscope[1])
    random_indices = np.random.choice(num_indices, int(source_indices_ratio * num_indices), replace=False)
    translation = find_translation_stochastic(curr_microscope[1], curr_well[1], random_indices)

    result[key] = { "translation": translation }
  
  with open(output_path, "wb") as file:
    pickle.dump(result, file)


@cli.command()
@click.option("--input_path", type=str, required=True, help="A valid path to read preprocessed microscope and biosensor data from.")
@click.option("--output_path", type=str, required=True, help="A valid path to to store results to.")
@click.option("--epsilon", type=str, required=True, help="Pairwise consistency threshold.")
def eval_pmc(input_path, output_path, epsilon):
  epsilon = float(epsilon)

  with open(input_path, "rb") as file:
    data = pickle.load(file)

  result = {}
  for key in data.keys():
    curr_well = data[key]["well_data"]
    curr_microscope = data[key]["microscope_data"]
    translation = find_translation_pmc(curr_microscope[1], curr_well[1], epsilon)

    result[key] = { "translation": translation }
  
  with open(output_path, "wb") as file:
    pickle.dump(result, file)


if __name__ == "__main__":
  cli()