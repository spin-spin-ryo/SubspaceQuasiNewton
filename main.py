from environments import * 
from numerical_experiment import get_objects_from_config
from utils.save_func import get_path_form_params,save_result_json,load_config,plot_results
import os
import sys
from utils.logger import logger
import numpy as np
PROXIMAL_METHODS = [PROXIMAL_GRADIENT_DESCENT,ACCELERATED_PROXIMAL_GRADIENT_DESCENT]



def run_numerical_experiment(config):
  iteration = config["iteration"]
  log_interval = config["log_interval"]
  algorithms_config = config["algorithms"]
  objectives_config = config["objective"]
  constraints_config = config["constraints"]
  solver_name = algorithms_config["solver_name"]
  objective_name = objectives_config["objective_name"]
  constraints_name = constraints_config["constraints_name"]

  use_prox = solver_name in PROXIMAL_METHODS

  solver,solver_params,f,function_properties,con,constraints_properties,x0, prox = get_objects_from_config(config)
  f.set_type(DTYPE)
  x0 = x0.astype(DTYPE)
  logger.info(f"dimensiton:{f.get_dimension()}")
  
  
  solver_dir = get_path_form_params(solver_params)
  func_dir = get_path_form_params(function_properties)
  
  if constraints_name != NOCONSTRAINTS:
    con_dir = get_path_form_params(constraints_properties)
    save_path = os.path.join(RESULTPATH,
                            objective_name,func_dir,
                            constraints_name,con_dir,
                            solver_name,solver_dir)
    con.set_type(DTYPE)
    if con.is_feasible(x0):
      logger.info("Initial point is feasible.")
    else:
      logger.info("Initial point is not feasible")
      return
  else:
    save_path = os.path.join(RESULTPATH,
                            objective_name,func_dir,
                            constraints_name,
                            solver_name,solver_dir)

  os.makedirs(save_path,exist_ok=True)
  logger.info(save_path)
  # 実験開始
  logger.info("Run Numerical Experiments")
  if constraints_name != NOCONSTRAINTS:
    if use_prox:
      solver.run(f=f,
                 prox=prox,
                 x0=x0,
                 iteration=iteration,
                 params=solver_params,
                 save_path=save_path,
                 log_interval=log_interval)
    else:
      solver.run(f=f,
                con=con,
                x0=x0,
                iteration=iteration,
                params=solver_params,
                save_path=save_path,
                log_interval=log_interval
                )
  else:
    solver.run(f=f,
              x0=x0,
              iteration=iteration,
              params=solver_params,
              save_path=save_path,
              log_interval=log_interval
              )

  nonzero_index = solver.save_values["func_values"] != 0
  min_f_value = np.min(solver.save_values["func_values"][nonzero_index])
  execution_time = solver.save_values["time"][-1]
  values_dict = {
    "min_value":min_f_value,
    "time":execution_time
  }
  plot_results(save_path,solver.save_values)

  save_result_json(save_path=os.path.join(save_path,"result.json"),
                  values_dict=values_dict,
                  iteration=iteration)
  logger.info("Finish Numerical Experiment")
if __name__ == "__main__":
  args = sys.argv
  config_path = args[1]
  config = load_config(config_path)
  run_numerical_experiment(config)