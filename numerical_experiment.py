from algorithms.solver import get_solver
from problems.generate_problem import generate_objective,generate_constraints,generate_initial_points,objective_properties_key,constraints_properties_key
from environments import NOCONSTRAINTS,DTYPE,REGULARIZED
from utils.calculate import identity_prox

def get_objects_from_config(config):
    algorithms_config = config["algorithms"]
    objectives_config = config["objective"]
    constraints_config = config["constraints"]

    # objectiveを取得
    objective_name = objectives_config["objective_name"]
    function_properties = {}
    if REGULARIZED in objective_name:
      function_properties["ord"] = objectives_config["ord"]
      function_properties["coeff"] = objectives_config["coeff"]
      function_properties["Fused"] = objectives_config["Fused"]
    
    for param in objective_properties_key[objective_name.replace(REGULARIZED,"")]:
      function_properties[param] = objectives_config[param]
      
    f = generate_objective(function_name=objective_name,function_properties=function_properties)
    if "dim" in function_properties:
      function_properties["dim"] = f.get_dimension()
    

    # solverを取得
    solver_name = algorithms_config["solver_name"]
    backward_mode = algorithms_config["backward"]
    solver = get_solver(solver_name=solver_name,dtype=DTYPE)
    solver_params = {}
    for param in solver.params_key:
      solver_params[param] = algorithms_config[param]
    if "dim" in solver_params:
      solver_params["dim"] = f.get_dimension()
          
    # constraintsを取得
    constraints_name = constraints_config["constraints_name"]
    constraints_properties = {}
    if constraints_name != NOCONSTRAINTS:
      for param in constraints_properties_key[constraints_name]:
        constraints_properties[param] = constraints_config[param]
      if "dim" in constraints_properties:
        constraints_properties["dim"] = f.get_dimension()
      con,prox = generate_constraints(constraints_name=constraints_name,constraints_properties=constraints_properties)
    else:
      con = None
      prox = identity_prox
    
        
    x0 = generate_initial_points(func=f,
                                 function_name=objective_name,
                                 constraints_name=constraints_name,
                                 function_properties=function_properties)
    
    return solver,solver_params,f,function_properties,con,constraints_properties,x0, prox


        