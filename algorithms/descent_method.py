import time
import numpy as np
import jax.numpy as jnp
from jax.lax import transpose
from jax import grad,jit
from utils.calculate import line_search,subspace_line_search,get_minimum_eigenvalue,hessian,jax_randn
from utils.logger import logger
import os
from environments import FINITEDIFFERENCE,DIRECTIONALDERIVATIVE

class optimization_solver:
  def __init__(self,dtype = jnp.float64) -> None:
    self.f = None 
    self.f_grad = None
    self.xk = None
    self.dtype = dtype
    self.backward_mode = True
    self.finish = False
    self.save_values = {}
    self.params_key = {}
    pass

  def __zeroth_order_oracle__(self,x):
    return self.f(x)

  def __first_order_oracle__(self,x,output_loss = False):
    if self.backward_mode:
      x_grad = self.f_grad(x)
      if output_loss:
        return x_grad,self.f(x)
      else:
        return x_grad
    
    elif isinstance(self.backward_mode,str):
      if self.backward_mode == DIRECTIONALDERIVATIVE:
        pass
      elif self.backward_mode == FINITEDIFFERENCE:
        pass
      else:
        raise ValueError(f"{self.backward_mode} is not implemented.")
      
  
  def __second_order_oracle__(self,x):
    return hessian(self.f)(x)
    
   
  def __clear__(self):
    return
  
  def __run_init__(self,f,x0,iteration):
    self.f = f
    self.f_grad = grad(self.f)
    self.xk = x0.copy()
    self.save_values["func_values"] = np.zeros(iteration+1)
    self.save_values["time"] = np.zeros(iteration+1)
    self.finish = False
    self.save_values["func_values"][0] = self.f(self.xk)

  def __check_params__(self,params):
    all_params = True
    assert len(self.params_key) == len(params),"不要,または足りないparamがあります."
    for param_key in self.params_key:
      if param_key in params:
        continue
      else:
        all_params &= False
    
    assert all_params, "パラメータが一致しません"
  
  def check_norm(self,d,eps):
    return jnp.linalg.norm(d) <= eps
  
  def run(self,f,x0,iteration,params,save_path,log_interval = -1):
    self.__run_init__(f,x0,iteration)
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
  
  def update_save_values(self,iter,**kwargs):
    for k,v in kwargs.items():
      self.save_values[k][iter] = v
  
  def save_results(self,save_path):
    for k,v in self.save_values.items():
      jnp.save(os.path.join(save_path,k+".npy"),v)
  
  def __update__(self,d):
    self.xk += d

  def __iter_per__(self,params):
    return

  def __direction__(self,grad):
    return
  
  def __step_size__(self,params):
    return
  
class GradientDescent(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = ["lr",
                       "backward"]
  
  def __iter_per__(self,params):
    grad = self.__first_order_oracle__(self.xk)
    d = self.__direction__(grad,params)
    alpha = self.__step_size__(d,params)
    self.__update__(alpha*d)
    return
  
  def __direction__(self,grad,params):
    return -grad
  
  def __step_size__(self,direction,params):
    return params["lr"]

class SubspaceGD(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = ["lr",
                       "reduced_dim",
                       "dim",
                       "mode",
                       "backward"]
        
  def subspace_first_order_oracle(self,x,Mk):
    reduced_dim = Mk.shape[0]
    subspace_func = lambda d:self.f(x + transpose(Mk,(1,0))@d)
    if self.backward_mode:
      d = jnp.zeros(reduced_dim,dtype=self.dtype)
      return grad(subspace_func)(d)
  
  def __iter_per__(self, params):
    reduced_dim = params["reduced_dim"]
    dim = params["dim"]
    mode = params["mode"]
    Mk = self.generate_matrix(dim,reduced_dim,mode)
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk)
    d = self.__direction__(projected_grad,Mk)
    alpha = self.__step_size__(params)
    self.__update__(alpha*d)

  def __direction__(self, projected_grad,Mk):
    return -Mk.transpose(0,1)@projected_grad
    
  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/dim
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")

class AcceleratedGD(optimization_solver):
  def __init__(self,  dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.lambda_k = 0
    self.yk = None
    self.params_key = ["lr",
                       "backward"]
  
  def __run_init__(self, f, x0, iteration):
    self.yk = x0.copy()
    return super().__run_init__(f, x0, iteration)
  
  def __iter_per__(self, params):
    lr = params["lr"]
    lambda_k1 = (1 + (1 + 4*self.lambda_k**2)**(0.5))/2
    gamma_k = ( 1 - self.lambda_k)/lambda_k1
    grad = self.__first_order_oracle__(self.xk)
    yk1 = self.xk - lr*grad
    self.xk = (1 - gamma_k)*yk1 + gamma_k*self.yk
    self.yk = yk1
    self.lambda_k = lambda_k1
        
class NewtonMethod(optimization_solver):
  def __init__(self,  dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = [
      "alpha",
      "beta",
      "backward"
    ]

  def __iter_per__(self, params):
    grad = self.__first_order_oracle__(self.xk)
    H = self.__second_order_oracle__(self.xk)
    dk = self.__direction__(grad=grad,hess=H)
    lr = self.__step_size__(grad=grad,dk=dk,params=params)
    self.__update__(lr*dk)
        
  def __direction__(self, grad,hess):
    return - jnp.linalg.solve(hess,grad)
    
  def __step_size__(self, grad,dk,params):
    alpha = params["alpha"]
    beta = params["beta"]
    return line_search(self.xk,self.f,grad,dk,alpha,beta)

class SubspaceNewton(SubspaceGD):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key =["dim",
                      "reduced_dim",
                      "mode",
                      "backward"]

  def subspace_second_order_oracle(self,x,Mk):
    reduced_dim = Mk.shape[0]
    d = jnp.zeros(reduced_dim,dtype = self.dtype)
    sub_func = lambda d: self.f(x +Mk.transpose(0,1)@d)
    return hessian(sub_func)(d)
    

  def __iter_per__(self, params):
    reduced_dim = params["reduced_dim"]
    dim = params["dim"]
    mode = params["mode"]
    Mk = self.generate_matrix(dim,reduced_dim,mode)
    grad = self.subspace_first_order_oracle(self.xk,Mk)
    H = self.subspace_second_order_oracle(self.xk,Mk)
    dk = self.__direction__(grad=grad,hess=H)
    lr = self.__step_size__(grad=grad,dk=dk,params=params,Mk=Mk)
    self.__update__(lr*Mk.transpose(0,1)@dk)
  
  def __direction__(self, grad,hess,Mk):
    return - jnp.linalg.solve(hess,grad)
    
  def __step_size__(self, grad,dk,Mk,params):
    alpha = params["alpha"]
    beta = params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)

  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/dim
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")

class LimitedMemoryNewton(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.Pk = None
    self.params_key = [
      "matrix_size",
      "threshold_eigenvalue",
      "alpha",
      "beta",
      "backward"
    ]
  
  def subspace_first_order_oracle(self,x,Mk):
    subspace_func = lambda d:self.f(x + Mk.transpose(0,1)@d)
    if self.backward_mode:
      matrix_size = Mk.shape[0]
      d = jnp.zeros(matrix_size,dtype=self.dtype)
      return grad(subspace_func)(d)
    
  def generate_matrix(self,matrix_size,gk):
    # P^\top = [x_0,\nabla f(x_0),...,x_k,\nabla f(x_k)]
    if self.Pk is None:
      self.Pk = jnp.concatenate([jnp.expand_dims(self.xk,0),jnp.expand_dims(gk,0)])
    else:
      if self.Pk.shape[0] < matrix_size:
        self.Pk = jnp.concatenate([self.Pk,jnp.expand_dims(self.xk,0),jnp.expand_dims(gk,0)])
      else:
        self.Pk = jnp.concatenate([self.Pk[2:],jnp.expand_dims(self.xk,0),jnp.expand_dims(gk,0)])

  def subspace_second_order_oracle(self,x,Mk,threshold_eigenvalue):
    matrix_size = Mk.shape[0]
    d = jnp.zeros(matrix_size,dtype = self.dtype)
    sub_loss = lambda d:self.f(x + Mk.transpose(0,1)@d)
    H = hessian(sub_loss)(d)
    sigma_m = get_minimum_eigenvalue(H)
    if sigma_m < threshold_eigenvalue:
        return H + (threshold_eigenvalue - sigma_m)*jnp.eye(matrix_size,dtype = self.dtype)
    else:
        return H                                                                                                                                                                                                                                                                                        
  
  def __iter_per__(self, params):
    matrix_size = params["matrix_size"]
    threshold_eigenvalue = params["threshold_eigenvalue"]
    proj_gk = self.subspace_first_order_oracle(self.xk,self.Pk)
    Hk = self.subspace_second_order_oracle(self.xk,self.Pk)
    dk = self.__direction__(grad=proj_gk,hess = Hk)
    lr = self.__step_size__(grad=proj_gk,dk=dk,Mk = self.Pk,params=params)
    self.__update__(lr*self.Pk.transpose(0,1)@dk)
  
  def __direction__(self, grad,hess):
    return - jnp.linalg.solve(hess,grad)
  
  def __step_size__(self, grad,dk,Mk,params):
    alpha = params["alpha"]
    beta = params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)

# prox(x,t):
class BacktrackingProximalGD(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.prox = None
    self.params_key = [
      "eps",
      "beta",
      "backward"
    ]
  
  def __run_init__(self, f, prox, x0, iteration):
    self.prox = prox
    return super().__run_init__(f, x0, iteration)
  
  def run(self, f, prox, x0, iteration, params,save_path,log_interval=-1):
    self.__run_init__(f,prox, x0,iteration)
    self.backward_mode = params["backward"]
    self.__check_params__(params)
    start_time = time.time()
    for i in range(iteration):
      self.__clear__()
      if not self.finish:
        self.__iter_per__(params)
      else:
        break
      self.save_values["time"][i+1] = time.time() - start_time
      self.save_values["func_values"][i+1] = self.f(self.xk)
      if (i+1)%log_interval == 0 & log_interval != -1:
        logger.info(f'{i+1}: {self.save_values["func_values"][i+1]}')
        self.save_results(save_path)
    return
  

  def backtracking_with_prox(self,x,grad,beta,max_iter = 10000,loss = None):
    t = 1
    if loss is None:
      loss = self.f(x)
    prox_x = self.prox(x - t*grad,t)
    while t*self.f(prox_x) > t*loss - t*grad@(x - prox_x) + 1/2*((x-prox_x)@(x-prox_x)):
      t *= beta
      max_iter -= 1
      prox_x = self.prox(x - t*grad,t)
      if max_iter < 0:
        logger.info("Error: Backtracking is stopped because of max_iteration.")
        break
    return prox_x,t
  
  def __iter_per__(self, params):
    beta = params["beta"]
    eps = params["eps"]
    grad,loss = self.__first_order_oracle__(self.xk,output_loss=True)
    prox_x,t = self.backtracking_with_prox(self.xk,grad,beta,loss=loss)
    if self.check_norm(self.xk - prox_x,t*eps):
      self.finish = True
    self.xk = prox_x.copy()     
    return

class BacktrackingAcceleratedProximalGD(BacktrackingProximalGD):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.tk = 1
    self.vk = None
    self.k = 0
    self.xk1 = None
    self.params_key = [
      "restart",
      "beta",
      "eps",
      "backward"
    ]
  
  def __run_init__(self,f, prox,x0,iteration):
    self.k = 0
    self.xk1 = x0.copy()
    return super().__run_init__(f,prox,x0,iteration)

  def __iter_per__(self, params):
    self.k+=1
    beta = params["beta"]
    eps = params["eps"]
    restart = params["restart"]
    k = self.k
    self.vk = self.xk + (k-2)/(k+1)*(self.xk - self.xk1)
    grad_v,loss_v = self.__first_order_oracle__(self.vk,output_loss=True)
    prox_x,t = self.backtracking_with_prox(self.xk,self.vk,grad_v,beta,loss_v)
    if self.check_norm(prox_x - self.vk,t*eps):
      self.finish = True
    self.xk1 = self.xk
    self.xk = prox_x.copy()
    self.v = None
    if restart:
      if self.f(self.xk) > self.f(self.xk1):
          self.k = 0

  def backtracking_with_prox(self, x,v, grad_v, beta, max_iter=10000, loss_v=None):
    if loss_v is None:
      loss_v = self.f(v)
    prox_x = self.prox(v-self.tk*grad_v,self.tk)
    while self.tk*self.f(prox_x) > self.tk*loss_v + self.tk*grad_v@(prox_x - v) + 1/2*((prox_x - v)@(prox_x - v)):
        self.tk *= beta
        prox_x = self.prox(v-self.tk*grad_v,self.tk)    
    return prox_x,self.tk
