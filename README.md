# Subspace quasi newton method

## How to run numerical experiments
1. Select optimization problem and an algorithm and set the parameters in `make_config.py`.
2. Make config.json file using command `python make_config.py`.
3. Run numerical experiments using command `python main.py path/to/config.json`.
4. Check the results in `results/problem_name/problem_parameters/constraints_name/constraints_parameters/algorithm_name/algorithm_parameters` directory.
5. You can compare results using `python result_show.py`. with GUI interface.

## problems

### QUADRATIC
$$\min_x \frac{1}{2}x^\top A x + b^\top x$$