from utils.save_func import save_config
from environments import *

save_path = "configs/config.json"

# select objective function name, constraints name and algorithm name.
# please refer to environment.py for the specific variable names.
objective_name = QUADRATIC
constraints_name = NOCONSTRAINTS
solver_name = GRADIENT_DESCENT

# The relevant parameters differ depending on the problem and method. 
# parameters of optimization problem
dim:int = 1000
constraints_num:int = 0
convex:bool = False
data_name:str = "random"
rank:int = 0
threshold1:int = 0
threshold2:int = 0
activation:str = "relu"
criterion:str = "CrossEntropy"

layers_size:list = [
  (1,16,5,1),
  (16,32,5,1)
]
ord:int = 0
threshold:int = 0

# parameters of algorithms
backward_mode:bool = True
iteration:int = 1000
log_interval:int = 1000
eps0:float = 1e-4
delta1:float = 1e-10
eps2:float = 1e-10
reduced_dim:int = 100
alpha1:float = 1e+10
alpha2:float = 1e+9
beta:float = 0.8
mode:str = "random" 
r:float = 0.5
alpha = 0.3
eps = 1e-4
delta = 1e-4
restart = True
mu = 1.5
eps_feas = 1e-4
lr = 0.1

if __name__ == "__main__":
  config = {
    "objective":{
      "objective_name":objective_name,
      "dim":dim,
      "convex":convex,
      "data_name":data_name,
      "rank":rank,
      "activation":activation,
      "criterion":criterion,
      "layers_size":layers_size
    },

    "constraints":{
      "constraints_name":constraints_name,
      "constraints_num":constraints_num,
      "dim":dim,
      "data_name":data_name,
      "threshold1":threshold1,
      "threshold2":threshold2,
      "ord":ord,
      "threshold":threshold
    },
    "algorithms":{
      "solver_name":solver_name,
      "backward":backward_mode,
      "dim":dim,
      "eps0":eps0,
      "delta1":delta1,
      "eps2":eps2,
      "reduced_dim":reduced_dim,
      "alpha1":alpha1,
      "alpha2":alpha2,
      "beta":beta,
      "mode":mode,
      "alpha":alpha,
      "eps":eps,
      "delta":delta,
      "restart":restart,
      "r":r,
      "mu":mu,
      "eps_feas":eps_feas,
      "lr":lr
    },
    "iteration":iteration,
    "log_interval":log_interval
  }

  save_config(config=config,save_path=save_path)
