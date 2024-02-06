from utils.save_func import save_config
from environments import *

save_path = "configs/config.json"

# 以下設定
objective_name = QUADRATIC
constraints_name = NOCONSTRAINTS
solver_name = GRADIENT_DESCENT

# 問題関連のパラメータ
dim = 1000
constraints_num = 0
convex = False
data_name = "random"
rank = 0
threshold1 = 0
threshold2 = 0
activation = "relu"
criterion = "CrossEntropy"

layers_size = [
  (1,16,5,1),
  (16,32,5,1)
]
ord = 0
threshold = 0

# アルゴリズム関連のパラメータ
backward_mode = True
iteration = 1000
log_interval = 1000
eps0 = 1e-4
delta1 = 1e-10
eps2 = 1e-10
reduced_dim = 100
alpha1 = 1e+10
alpha2 = 1e+9
beta = 0.8
mode = "random" 
r = 0.5

alpha = 0.3
eps = 1e-4
delta = 1e-4
restart = True
mu = 1.5
eps_feas = 1e-4
lr = 0.1

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
