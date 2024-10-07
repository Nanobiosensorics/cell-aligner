import numpy as np
import time
import methods
import click
import pickle
from memory_profiler import memory_usage


@click.group()
def cli():
  """Main entry point for the CLI."""
  pass


@cli.command()
@click.option("--n_source_points_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--n_target_points_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--n_source_indices_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--out_path", type=str, required=True, help="A valid path to store results.")
def eval_stochastic(n_source_points_array: str, n_target_points_array: str, n_source_indices_array: str, out_path: str):
  n_source_points_array = __parse_1d_int_array(n_source_points_array)
  n_target_points_array = __parse_1d_int_array(n_target_points_array)
  n_source_indices_array = __parse_1d_int_array(n_source_indices_array)
  assert len(n_source_points_array) == len(n_target_points_array) == len(n_source_indices_array)

  output = []
  for i in range(len(n_source_points_array)):
    output.append({
      "input": {
        "n_source_points": n_source_points_array[i],
        "n_target_points": n_target_points_array[i],
        "n_source_indices": n_source_indices_array[i]
      },
      "result": __benchmark_stochastic(n_source_points_array[i], n_target_points_array[i], n_source_indices_array[i])
    })
    
    with open(out_path, 'wb') as file:
      pickle.dump(output, file)


@cli.command()
@click.option("--n_source_points_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--n_target_points_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--epsilon_array", type=str, required=True, help="1D array as a comma-separated string (e.g. 1,2,3).")
@click.option("--out_path", type=str, required=True, help="A valid path to store results.")
def eval_pmc(n_source_points_array: str, n_target_points_array: str, epsilon_array: str, out_path: str):
  n_source_points_array = __parse_1d_int_array(n_source_points_array)
  n_target_points_array = __parse_1d_int_array(n_target_points_array)
  epsilon_array = __parse_1d_float_array(epsilon_array)
  assert len(n_source_points_array) == len(n_target_points_array) == len(epsilon_array)

  output = []
  for i in range(len(n_source_points_array)):
    output.append({
      "input": {
        "n_source_points": n_source_points_array[i],
        "n_target_points": n_target_points_array[i],
        "epsilon": epsilon_array[i]
      },
      "result": __benchmark_pmc(n_source_points_array[i], n_target_points_array[i], epsilon_array[i])
    })

  with open(out_path, 'wb') as file:
    pickle.dump(output, file)   


# Parses a command line string input as an int array.
def __parse_1d_int_array(str):
  array_1d = list(map(int, str.split(',')))
  return array_1d


# Parses a command line string input as a float array.
def __parse_1d_float_array(str):
  array_1d = list(map(float, str.split(',')))
  return array_1d


# Benchmarks an evaluation method by measuring execution time and maximal memory usage.
def __benchmark_method(method_fn, params):
  start_time = time.time()
  max_usage = memory_usage((method_fn, params), max_usage=True)
  end_time = time.time()

  return (end_time - start_time), max_usage


# Benchmark the stochastic method implementation by generating random input.
def __benchmark_stochastic(n_source_points: int, n_target_points: int, n_source_indices: int):
  assert n_source_indices <= n_source_points
  np.random.seed(42)

  rnd_source_points = np.random.rand(n_source_points, 2)
  rnd_target_points = np.random.rand(n_target_points, 2)
  rnd_source_indices = np.random.randint(0, n_source_points, size=(n_source_indices))

  execution_time, max_memory = __benchmark_method(
    methods.find_translation_stochastic,
    (rnd_source_points, rnd_target_points, rnd_source_indices)
  )

  return {
    "execution_time": execution_time,
    "max_memory": max_memory
  }


# Benchmark the algorithms which finds the maximal cliques. (pmc)
def __benchmark_pmc(n_source_points: int, n_target_points: int, epsilon: float):
  np.random.seed(42)

  rnd_source_points = np.random.rand(n_source_points, 2)
  rnd_target_points = np.random.rand(n_target_points, 2)

  execution_time, max_memory = __benchmark_method(
    methods.find_translation_pmc,
    (rnd_source_points, rnd_target_points, epsilon)
  )

  return {
    "execution_time": execution_time,
    "max_memory": max_memory
  }


if __name__ == "__main__":
  cli()