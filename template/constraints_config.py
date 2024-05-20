from environments import *

def get_constraints_config(constraints_name,allparams_config = None):
  config = {}
  config["constraints_name"] = constraints_name
  if allparams_config is not None:
    for key in constraints_properties_key[constraints_name]:
      config[key] = allparams_config[key]
    return config
  if constraints_name == POLYTOPE:
    config["data_name"] = "random"
    config["dim"] = 1000
    config["constraints_num"] = 100
  elif constraints_name == NONNEGATIVE:
    config["dim"] = 1000
  elif constraints_name == QUADRATIC:
    config["data_name"] = "random"
    config["dim"] = 1000
    config["constraints_num"] = 10
  elif constraints_name == FUSEDLASSO:
    config["threshold1"] = 12000
    config["threshold2"] = 146000
  elif constraints_name == BALL:
    config["ord"] = 2
    config["threshold"] = 12000
  elif constraints_name == HUBER:
    config["delta"] =1
    config["threshold"] = 10
  return config