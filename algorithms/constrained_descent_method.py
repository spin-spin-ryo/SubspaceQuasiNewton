import time
import numpy as np
from algorithms.descent_method import optimization_solver,BacktrackingAcceleratedProximalGD
from utils.calculate import nonnegative_projection
from utils.logger import logger
import jax.numpy as jnp
from jax.lax import transpose

BARRIERTYPE1 = "values"
BARRIERTYPE2 = "grads" 

class constrained_optimization_solver(optimization_solver):
  def __init__(self,dtype = jnp.float64) -> None:
    super().__init__(dtype)
    self.con = None
  
  def run(self, f, con, x0, iteration, params, save_path, log_interval=-1):
    self.__run_init__(f,con, x0, iteration)
    self.__check_params__(params)
    self.backward_mode = params["backward"]
    start_time = time.time()
    for i in range(iteration):
      self.__clear__()
      if not self.finish:
        self.__iter_per__(params)
      else:
        logger.info("Stop Criterion")
        break
      T = time.time() - start_time
      F = self.f(self.xk)
      self.update_save_values(i+1,time = T,func_values = F)
      if (i+1)%log_interval == 0 & log_interval != -1:
        logger.info(f'{i+1}: {self.save_values["func_values"][i+1]}')
        self.save_results(save_path)
    return
  
  def __run_init__(self, f,con, x0, iteration):
    self.f = f
    self.con = con
    self.xk = x0.copy()
    self.save_values["func_values"] = np.zeros(iteration+1)
    self.save_values["time"] = np.zeros(iteration+1)
    self.finish = False
    self.save_values["func_values"][0] = self.f(self.xk)


  def evaluate_constraints_values(self,x):
    return self.con(x)
  
  def evaluate_constraints_grads(self,x):
    return self.con.grad(x)
  
class GradientProjectionMethod(constrained_optimization_solver):
  def __init__(self,dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = ["eps","delta","alpha","beta","backward"]
    self.lk = None
    # eps: active set
    # delta: gradient norm
    # alpha: stepsize
    # beta: linesearch

  def get_activate_grads(self,eps):
    # output: (*,n)
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    activate_constriants_index = constraints_values > - eps
    return constraints_grads[activate_constriants_index]

  def __iter_per__(self, params:dict):
    # 有効制約を保持しながら更新するともう少し高速になる
    eps = params["eps"]
    delta = params["delta"]
    alpha = params["alpha"]
    beta = params["beta"]
    # (*,n)
    Gk = self.get_activate_grads(eps) 
    grad = self.__first_order_oracle__(self.xk)
    d = self.__direction__(grad,Gk)
    if jnp.linalg.norm(d) < delta:
      if jnp.min(self.lk) >=0 :
        self.finish = True
        return
      else: 
        use_index = np.ones(self.lk.shape[0],dtype = bool)
        index_minus_element = jnp.argmin(self.lk)
        use_index[index_minus_element] = False
        Gk = Gk[use_index]
        d = self.__direction__(grad,Gk)
    
    alpha = self.__step_size__(d,alpha,beta)
    self.__update__(alpha*d)

  def __step_size__(self, direction,alpha,beta):
    while not self.con.is_feasible(self.xk + alpha*direction):
      alpha *= beta
    return alpha

  def __direction__(self,grad,Gk):
    if len(Gk) != 0:
      GG = Gk@transpose(Gk,(1,0))
      self.lk = - jnp.linalg.solve(GG,Gk@grad)
      return -grad - transpose(Gk,(1,0))@self.lk
    else:
      return -grad
    
class DynamicBarrierGD(constrained_optimization_solver):
  def __init__(self,dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.lk = None
    self.params_key = [
      "lr",
      "alpha",
      "beta",
      "barrier_func_type",
      "sub_problem_eps",
      "inner_iteration",
      "backward"
    ]   
  
  def barrier_func(self,constraints_grads,constraints_values,alpha,beta,type):
    # output [\phi_i(x)]_iを出力.
    if type == BARRIERTYPE1:
      return alpha*constraints_values
    elif type == BARRIERTYPE2:
      return beta*jnp.linalg.norm(constraints_grads,axis = 1)**2
  
  def get_lambda(self,grad,constraints_grads,constraints_values,alpha,beta,type,sub_problem_eps = 1e-6,inner_iteration = 100000):
    if constraints_grads.shape[0] == 1:
      # 制約が一つの場合
      l = (self.barrier_func(constraints_grads,constraints_values,alpha,beta,type) - grad@constraints_grads)/(jnp.linalg.norm(constraints_grads)**2)
      if l >= 0:
        self.lk = l
      else:
        self.lk = jnp.zeros(1,dtype = self.dtype)
    else:
      barrier_func_values = self.barrier_func(constraints_grads,constraints_values,alpha,beta,type)
      def func(l):
        return 1/2*(grad + transpose(constraints_grads,(1,0))@l)@(grad + transpose(constraints_grads,(1,0))@l) - l@barrier_func_values
      self.lk = self.solve_subproblem_by_APGD(func,nonnegative_projection,sub_problem_eps,inner_iteration)



  def solve_subproblem_by_APGD(self,func,prox,x0,sub_problem_eps,inner_iteration):
    solver = BacktrackingAcceleratedProximalGD(dtype = self.dtype)
    params = {"restart":True,
              "beta":0.8,
              "eps":sub_problem_eps}
    solver.run(x0=x0,
               f=func,
               prox=prox,
               iteration=inner_iteration,
               params=params)
    return solver.xk
    

  def __iter_per__(self, params):
    alpha = params["alpha"]
    beta = params["beta"]
    lr = params["lr"]
    barrier_func_type = params["barrier_func_type"]
    sub_problem_eps = 1e-6
    inner_iteration = 10000
    grad = self.__first_order_oracle__(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    constraints_values = self.evaluate_constraints_values(self.xk)
    self.get_lambda(grad,constraints_grads,constraints_values,alpha,beta,barrier_func_type,sub_problem_eps,inner_iteration)
    d = self.__direction__(grad,constraints_grads)
    self.__update__(lr*d)
  
  def __direction__(self, grad,constraints_grads):
    return - grad - transpose(constraints_grads,(1,0))@self.lk

class PrimalDualInteriorPointMethod(constrained_optimization_solver):
  def __init__(self,dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.lk = None
    self.params_key = [
      "mu",
      "eps",
      "eps_feas",
      "beta", #0.8
      "alpha", # 0.3
      "backward"
    ]
  
  def evaluate_constraints_hessian_linear_sum(self, x,l):
    return self.con.second_order_oracle(x,l)
  
  def get_surrogate_duality_gap(self,constraints_values):
    return - constraints_values@self.lk
  
  def __run_init__(self, f, con, x0, iteration):
    m = con.get_number_of_constraints()
    self.lk = jnp.ones(m,dtype = self.dtype,device = self.device)
    return super().__run_init__(f, con, x0, iteration)
  
  def __iter_per__(self, params):
    mu = params["mu"] # mu > 1
    eps = params["eps"]
    eps_feas = params["eps_feas"]
    m = self.con.get_number_of_constraints()
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    H = self.__second_order_oracle__(self.xk)
    grad = self.__first_order_oracle__(self.xk)
    constraints_hessian_linear_combination = self.evaluate_constraints_hessian_linear_sum(self.xk,self.lk)
    eta = self.get_surrogate_duality_gap(constraints_values)
    t = mu*m/eta
    delta_x,delta_l,r_t,r_dual = self.__direction__(t,grad,H,constraints_values,constraints_grads,constraints_hessian_linear_combination)
    if self.check_norm(r_dual,eps_feas) and eta<=eps:
      self.finish =True
      return
    s = self.__step_size__(delta_x,delta_l,r_t,t,params)
    self.__update__(s*delta_x,s*delta_l)

  def __update__(self, delta_x,delta_l):
    self.xk += delta_x
    self.lk += delta_l

  def get_r_dual(self,l,grad,constraints_grads):
    return grad + transpose(constraints_grads,(1,0))@l

  def get_r_cent(self,l,t,constraints_values):
    return - jnp.diag(l)@constraints_values - 1/t

  def get_r_t(self,l,t,grad,constraints_values,constraints_grads):
    r_dual = self.get_r_dual(l,grad,constraints_grads)
    r_cent = self.get_r_cent(l,t,constraints_values)
    r = jnp.concatenate([r_dual,r_cent])
    return r
  
  def __direction__(self, t,grad,H,constraints_values,constraints_grads,constraints_hessian_linear_combination):
    r_dual = self.get_r_dual(self.lk,grad,constraints_grads)
    r_cent = self.get_r_cent(self.lk,t,constraints_values)
    A11 = H + constraints_hessian_linear_combination
    A12 = constraints_grads.transpose(0,1)
    A21 = -jnp.diag(self.lk)@constraints_grads
    A22 = -jnp.diag(constraints_values)
    A_1 = jnp.concatenate([A11,A12],axis = 1)
    A_2 = jnp.concatenate([A21,A22],axis = 1)
    A = jnp.concatenate([A_1,A_2])
    r = jnp.concatenate([r_dual,r_cent])
    delta_y = jnp.linalg.solve(A,-r)
    delta_x = delta_y[:r_dual.shape[0]]
    delta_l = delta_y[r_dual.shape[0]:]
    return delta_x,delta_l,r,r_dual  
    
  def __step_size__(self, delta_x,delta_l,r_t,t,params):
    beta = params["beta"]
    alpha = params["alpha"]

    if jnp.all(delta_l>=0):
        s_max = 1
    else:
        s_max = jnp.min(jnp.array([1, jnp.min(-self.lk[delta_l<0]/delta_l[delta_l<0] )],dtype = self.dtype))
    s = 0.99*s_max
    
    while True:
      x = self.xk + s*delta_x
      l = self.lk + s*delta_l
      constraints_values = self.evaluate_constraints_values(x)
      if not jnp.all(constraints_values <= 0):
        s *= beta
        continue
      constraints_grads = self.evaluate_constraints_grads(x)
      grad = self.__first_order_oracle__(x)
      r_t_z = self.get_r_t(l,t,grad,constraints_values,constraints_grads)

      if not jnp.linalg.norm(r_t_z) <= (1 - alpha * s)*jnp.linalg.norm(r_t):
        s *= beta     
      break    
    return s
  
             
