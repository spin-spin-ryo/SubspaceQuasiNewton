from utils.save_func import save_config
from environments import *

save_path = "config_template.json"

# 以下設定
objective_name = CNN
constraints_name = BALL
solver_name = RSG_NC

# 問題関連のパラメータ
dim = 0
constraints_num = 100
convex = False
data_name = "mnist"
rank = 5
threshold1 = 12000
threshold2 = 146000
activation = "relu"
criterion = "CrossEntropy"

layers_size = [
  (1,16,5,1),
  (16,32,5,1)
]
ord = 2
threshold = 50

# アルゴリズム関連のパラメータ
backward_mode = True
iteration = 1
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
    "eps_feas":eps_feas
  },
  "iteration":iteration,
  "log_interval":log_interval
}

save_config(config=config,save_path=save_path)
