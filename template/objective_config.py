from environments import *

def  get_objective_config(objective_name,allparams_config = None):
  config = {}
  config["objective_name"] = objective_name
  if allparams_config is not None:
    if REGULARIZED in objective_name:
      for key in objective_properties_key[REGULARIZED]:
        config[key] = allparams_config[key] 
      objective_name = objective_name.replace(REGULARIZED,"")
    for key in objective_properties_key[objective_name]:
      config[key] = allparams_config[key]
    return config
  if REGULARIZED in objective_name:
    config["ord"] = 2
    config["coeff"] = 1e-4
    config["Fused"] = False
    objective_name = objective_name.replace(REGULARIZED,"")
    
  if objective_name == QUADRATIC:
    config["dim"] = 1000
    config["convex"] = True
    config["data_name"] = "random"
  elif objective_name == SPARSEQUADRATIC:
    config["dim"] = 1000
    config["data_name"] = "random"
  elif objective_name == MATRIXFACTORIZATION:
    config["data_name"] = "movie"
    config["rank"] = 5
  elif objective_name == MATRIXFACTORIZATION_COMPLETION:
    config["data_name"] = "movie"
    config["rank"] = 5
  elif objective_name == LEASTSQUARE:
    config["data_name"] = "random"
    config["data_size"] = 100
    config["dim"] = 1000
  elif objective_name == MLPNET:
    config["data_name"] = "mnist"
    config["layers_size"] =[
              (28*28,512,1),
              (512,512,1),
              (512,10,1)
            ]
    # config["layers_size"] =[
    #           (28*28,64,1),
    #           (64,64,1),
    #           (64,10,1)
    #         ]
    config["activation"] = "relu"
    config["criterion"] = "CrossEntropy"
  elif objective_name == CNN:
    config["data_name"] = "mnist"
    config["layers_size"] =[
              (1,16,5,1),
              (16,32,5,1)
            ]
    config["activation"] = "relu"
    config["criterion"] = "CrossEntropy"
  elif objective_name == SOFTMAX:
    config["data_name"] = "random"
  elif objective_name == LOGISTIC:
    config["data_name"] = "random"
  elif objective_name == SPARSEGAUSSIANPROCESS:
    config["data_name"] = "random"
    config["reduced_data_size"] = 100
    config["kernel_mode"] = "exp"
  
  return config