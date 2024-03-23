import time
from jax.numpy import float64
import numpy as np
import jax.numpy as jnp
from jax.lax import transpose
from jax import grad,jit
from utils.calculate import line_search,subspace_line_search,get_minimum_eigenvalue,hessian,jax_randn,get_jvp,get_hessian_with_hvp
from utils.logger import logger
import os
from environments import FINITEDIFFERENCE,DIRECTIONALDERIVATIVE,LEESELECTION,RANDOM

class optimization_solver:
  def __init__(self,dtype = jnp.float64) -> None:
    self.f = None 
    self.f_grad = None
    self.xk = None
    self.dtype = dtype
    self.backward_mode = True
    self.finish = False
    self.check_count = 0
    self.gradk_norm = None
    self.save_values = {}
    self.params_key = {}
    self.params = {}
    pass

  def __zeroth_order_oracle__(self,x):
    return self.f(x)

  def __first_order_oracle__(self,x,output_loss = False):
    x_grad = None
    if isinstance(self.backward_mode,str):
      if self.backward_mode == DIRECTIONALDERIVATIVE:
        x_grad = get_jvp(self.f,x,None)
      elif self.backward_mode == FINITEDIFFERENCE:
        dim = x.shape[0]
        d = np.zeros(dim,dtype=self.dtype)
        h = 1e-8
        e = np.zeros(dim,dtype=x.dtype)
        e[0] = 1
        e = jnp.array(e)
        z = self.f(x)
        for i in range(dim):
          d[i] = (self.f(x + h*e) - z)/h
        x_grad = jnp.array(d)
      else:
        raise ValueError(f"{self.backward_mode} is not implemented.")
    elif self.backward_mode:
      x_grad = self.f_grad(x)

    if output_loss:
      return x_grad,self.f(x)
    else:
      return x_grad
       
  def __second_order_oracle__(self,x):
    return hessian(self.f)(x)

  def subspace_first_order_oracle(self,x,Mk):
    reduced_dim = Mk.shape[0]
    if isinstance(self.backward_mode,str):
      if self.backward_mode == DIRECTIONALDERIVATIVE:
        return get_jvp(self.f,x,Mk)
      elif self.backward_mode == FINITEDIFFERENCE:
        d = np.zeros(reduced_dim,dtype=self.dtype)
        h = 1e-8
        z = self.f(x)
        for i in range(reduced_dim):
          d[i] = (self.f(x + h*Mk[i]) - z)/h
        return jnp.array(d)
    elif self.backward_mode:
      subspace_func = lambda d:self.f(x + Mk.T@d)
      d = jnp.zeros(reduced_dim,dtype=self.dtype)
      return grad(subspace_func)(d)
  
  def subspace_second_order_oracle(self,x,Mk):
    if isinstance(self.backward_mode,str):
      if self.backward_mode == DIRECTIONALDERIVATIVE:
        return get_hessian_with_hvp(self.f,x,Mk)
    else:
      reduced_dim = Mk.shape[0]
      d = jnp.zeros(reduced_dim,dtype = self.dtype)
      sub_func = lambda d: self.f(x +Mk.T@d)
      return hessian(sub_func)(d)
       
  def __clear__(self):
    return
  
  def __run_init__(self,f,x0,iteration,params):
    self.f = f
    self.f_grad = grad(self.f)
    self.xk = x0.copy()
    self.save_values["func_values"] = np.zeros(iteration+1)
    self.save_values["time"] = np.zeros(iteration+1)
    self.save_values["grad_norm"] = np.zeros(iteration+1)
    self.finish = False
    self.backward_mode = params["backward"]
    self.params = params
    self.check_count = 0
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
    d_norm = jnp.linalg.norm(d)
    self.update_save_values(self.check_count,grad_norm = d_norm)
    self.check_count +=1
    return d_norm <= eps
  
  def run(self,f,x0,iteration,params,save_path,log_interval = -1):
    self.__check_params__(params)
    self.__run_init__(f,x0,iteration,params)
    start_time = time.time()
    for i in range(iteration):
      self.__clear__()
      if not self.finish:
        self.__iter_per__()
      else:
        logger.info("Stop Criterion")
        break
      T = time.time() - start_time
      F = self.f(self.xk)
      self.update_save_values(i+1,time = T,func_values = F)
      if (i+1)%log_interval == 0 and log_interval != -1:
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

  def __iter_per__(self):
    return

  def __direction__(self,grad):
    return
  
  def __step_size__(self):
    return

# first order method
class GradientDescent(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = ["lr",
                       "eps",
                       "backward",
                       "linesearch"]
  
  def __iter_per__(self):
    grad = self.__first_order_oracle__(self.xk)
    if self.check_norm(grad,self.params["eps"]):
      self.finish = True
      return
    d = self.__direction__(grad)
    alpha = self.__step_size__(d,grad)
    self.__update__(alpha*d)
    return
  
  def __direction__(self,grad):
    return -grad
  
  def __step_size__(self,direction,grad):
    if self.params["linesearch"]:
      return line_search(xk = self.xk,
                         func = self.f,
                         grad = grad,
                         dk = direction,
                         alpha = 0.3,
                         beta = 0.8)
    else:
      return self.params["lr"]

class SubspaceGD(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = ["lr",
                       "reduced_dim",
                       "dim",
                       "mode",
                       "eps",
                       "backward",
                       "linesearch"]
        
  def __iter_per__(self):
    reduced_dim = self.params["reduced_dim"]
    dim =  self.params["dim"]
    mode =  self.params["mode"]
    Mk = self.generate_matrix(dim,reduced_dim,mode)
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk)
    if self.check_norm(projected_grad, self.params["eps"]):
      self.finish = True
    d = self.__direction__(projected_grad,Mk)
    alpha = self.__step_size__(direction=d,
                            projected_grad=projected_grad,
                            Mk = Mk)
    self.__update__(alpha*Mk.T@d)
  
  def __step_size__(self,direction,projected_grad,Mk):
    if self.params["linesearch"]:
      return subspace_line_search(xk = self.xk,
                                  func = self.f,
                                  projected_grad=projected_grad,
                                  dk = direction,
                                  Mk = Mk,
                                  alpha = 0.3,
                                  beta = 0.8)
    else:
      return self.params["lr"]

  def __direction__(self, projected_grad,Mk):
    return -projected_grad
    
  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/(reduced_dim**0.5)
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
                       "eps",
                       "backward",
                       "restart"]
  
  def __run_init__(self, f, x0, iteration,params):
    self.yk = x0.copy()
    return super().__run_init__(f, x0, iteration,params)
  
  def __iter_per__(self):
    lr = self.params["lr"]
    lambda_k1 = (1 + (1 + 4*self.lambda_k**2)**(0.5))/2
    gamma_k = ( 1 - self.lambda_k)/lambda_k1
    grad = self.__first_order_oracle__(self.xk)
    if self.check_norm(grad,self.params["eps"]):
      self.finish = True
      return
    yk1 = self.xk - lr*grad
    xk1 = (1 - gamma_k)*yk1 + gamma_k*self.yk
    if self.params["restart"]:
      if self.f(xk1) > self.f(self.xk):
        self.lambda_k = 0
        return
    self.xk = xk1
    self.yk = yk1
    self.lambda_k = lambda_k1

class BFGS(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = [
      "alpha",
      "beta",
      "backward",
      "eps"
    ]
    self.Hk = None
    self.gradk = None

  
  def __run_init__(self, f, x0, iteration,params):
    self.Hk = jnp.eye(x0.shape[0],dtype = self.dtype)
    super().__run_init__(f, x0, iteration,params)
    self.gradk = self.__first_order_oracle__(x0)
    self.check_norm(self.gradk,params["eps"])
    return 
  
  def __direction__(self, grad):
    return -self.Hk@grad
  
  def __step_size__(self, grad,dk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return line_search(self.xk,self.f,grad,dk,alpha,beta)

  def __iter_per__(self):
    dk = self.__direction__(self.gradk)
    s = self.__step_size__(grad=self.gradk,
                           dk = dk)
    self.__update__(s*dk)
    gradk1 = self.__first_order_oracle__(self.xk)
    if self.check_norm(gradk1,self.params["eps"]):
      self.finish = True
      return
    yk = gradk1 - self.gradk
    self.update_BFGS(sk = s*dk,yk = yk)
    self.gradk = gradk1

  def update_BFGS(self,sk,yk):
    # a ~ 0
    a = sk@yk
    if a < 1e-14:
      self.Hk = jnp.eye(sk.shape[0],dtype = self.dtype)
      return
    B = jnp.dot(jnp.expand_dims(self.Hk@yk,1),jnp.expand_dims(sk,0))
    S = jnp.dot(jnp.expand_dims(sk,1),jnp.expand_dims(sk,0))
    self.Hk = self.Hk + (a + self.Hk@yk@yk)*S/(a**2) - (B + B.T)/a

class LimitedMemoryBFGS(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.s = None
    self.y = None
    self.r = None
    self.a = None
    self.b = None
    self.gradk = None
    self.params_key = [
      "alpha",
      "beta",
      "backward",
      "eps",
      "memory_size"
    ]
  
  def __run_init__(self, f, x0, iteration, params):
    super().__run_init__(f, x0, iteration, params)
    m = params["memory_size"]
    dim = x0.shape[0]
    self.s = jnp.zeros((m,dim),dtype = self.dtype)
    self.y = jnp.zeros((m,dim),dtype = self.dtype)
    self.r = np.zeros(m,dtype = self.dtype)
    self.a = np.zeros(m,dtype = self.dtype)
    self.gradk = self.__first_order_oracle__(x0)
    self.check_norm(self.gradk,params["eps"])
    
  def __direction__(self, grad):
    g = grad
    memory_size = self.a.shape[0]
    param_reset = self.r[0] < 1e-14
    for i in range(memory_size):
      if param_reset:
        self.r[i] = 0
      if self.r[i] < 1e-14:
        self.a[i] = 0
        self.r[i] = 0
      else:
        self.a[i] = jnp.dot(self.s[i],g)/self.r[i]
        g -= self.a[i]*self.y[i]
    
    if param_reset:
      return - grad

    gamma = jnp.dot(self.s[0],self.y[0])/jnp.dot(self.y[0],self.y[0])
    z = gamma*g
    for i in range(1,memory_size+1):
      if self.r[-i] < 1e-14:
        continue
      else:
        b = jnp.dot(self.y[-i],z)/self.r[-i]
        z += self.s[-i]*(self.a[-i] - b)
    
    return -z

  def __iter_per__(self):
    dk = self.__direction__(self.gradk)
    s = self.__step_size__(grad=self.gradk,
                           dk = dk)
    self.__update__(s*dk)
    gradk1 = self.__first_order_oracle__(self.xk)
    if self.check_norm(gradk1,self.params["eps"]):
      self.finish = True
      return
    yk = gradk1 - self.gradk
    self.update_BFGS(sk = s*dk,yk = yk)
    self.gradk = gradk1
  
  def __step_size__(self, grad,dk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return line_search(self.xk,self.f,grad,dk,alpha,beta)

  def update_BFGS(self,sk,yk):
    self.s = jnp.roll(self.s,1,axis = 0)
    self.y = jnp.roll(self.y,1,axis=0)
    self.r = np.array(jnp.roll(self.r,1))
    self.s = self.s.at[0].set(sk)
    self.y = self.y.at[0].set(yk)
    self.r[0] = jnp.dot(sk,yk)

class AcceleratedGDRestart(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.k = 0
    self.K = 0
    self.L = 0
    self.Sk = 0
    self.yk = None
    self.grad_xk = None
    self.grad_yk = None
    self.initial_loss = None
    self.params_key = ["L","M","alpha","beta","backward"]

  def __run_init__(self, f, x0, iteration,params):
    super().__run_init__(f, x0, iteration,params)
    self.yk = x0.copy()
    self.k = 0
    self.L = self.params["L"]
    self.Sk = 0
    self.Mk = self.params["M"]
    self.grad_xk = self.__first_order_oracle__(self.xk)
    self.grad_yk = self.grad_xk.copy()
    self.initial_loss = self.save_values["func_values"][0]
  
  def update_Sk(self,xk1,xk):
    self.Sk += jnp.linalg.norm(xk1 - xk)**2
  
  def __iter_per__(self):
    self.k+=1
    xk1 = self.yk - self.grad_yk/self.L
    yk1 = self.xk + self.k/(self.k+1)*(xk1 - self.xk)
    self.update_Sk(xk1,self.xk)
    grad_xk1,loss_xk1 = self.__first_order_oracle__(xk1,output_loss=True)
    if loss_xk1 > self.initial_loss - self.L*self.Sk/(2*(self.k + 1)):
      self.restart_iter(self.xk,self.params["alpha"]*self.L,grad_x0 = self.grad_xk,grad_y0 = self.grad_yk)
      return
    
    grad_yk1,loss_yk1 = self.__first_order_oracle__(yk1,output_loss=True)
    self.update_Mk(loss_xk1=loss_xk1,
                   loss_yk1=loss_yk1,
                   grad_yk1=grad_yk1,
                   grad_xk1=grad_xk1,
                   grad_xk=self.grad_xk,
                   xk1 = xk1,
                   xk = self.xk,
                   yk1 = yk1,
                   theta_k = self.k/(self.k + 1))
    if (self.k+1)**5 * self.Mk**2 * self.Sk > self.L**2:
      self.restart_iter(xk1,self.params["beta"]*self.L,grad_x0 = grad_xk1,grad_y0 = grad_yk1)
      return
    self.xk = xk1
    self.yk = yk1
    self.grad_xk = grad_xk1
    self.grad_yk = grad_yk1
  
  def update_Mk(self,loss_xk1,loss_yk1,grad_yk1,grad_xk1,grad_xk,xk1,xk,yk1,theta_k):
    a = 12*(loss_yk1 - loss_xk1 - 0.5*jnp.dot(grad_yk1 + grad_xk1,yk1 -xk1))/ ( jnp.linalg.norm(yk1 - xk1)**3 )
    b = jnp.linalg.norm(grad_yk1 + theta_k*grad_xk - (1+ theta_k)*grad_xk1)/(theta_k*jnp.linalg.norm(xk1 - xk)**2)
    self.Mk = max(self.Mk,a,b)

  def restart_iter(self,x0,L,grad_x0,grad_y0):
    self.xk = x0.copy()
    self.yk = x0.copy()
    self.L = L
    self.k = 0
    self.grad_yk = grad_y0
    self.grad_xk = grad_x0
    

# prox(x,t):
class BacktrackingProximalGD(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.prox = None
    self.params_key = [
      "eps",
      "beta",
      "backward",
      "alpha"
    ]
  
  def __run_init__(self, f, prox, x0, iteration,params):
    self.prox = prox
    return super().__run_init__(f, x0, iteration,params)
  
  def run(self, f, prox, x0, iteration, params,save_path,log_interval=-1):
    self.__check_params__(params)
    self.__run_init__(f,prox, x0,iteration,params)
    start_time = time.time()
    for i in range(iteration):
      self.__clear__()
      if not self.finish:
        self.__iter_per__()
      else:
        break
      self.save_values["time"][i+1] = time.time() - start_time
      self.save_values["func_values"][i+1] = self.f(self.xk)
      if (i+1)%log_interval == 0 and log_interval != -1:
        logger.info(f'{i+1}: {self.save_values["func_values"][i+1]}')
        self.save_results(save_path)
    return
  

  def backtracking_with_prox(self,x,grad,beta,t = 1,max_iter = 10000,loss = None):
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
  
  def __iter_per__(self):
    beta = self.params["beta"]
    eps = self.params["eps"]
    alpha = self.params["alpha"]
    grad,loss = self.__first_order_oracle__(self.xk,output_loss=True)
    prox_x,t = self.backtracking_with_prox(self.xk,grad,beta,t =alpha,loss=loss)
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
      "backward",
      "alpha"
    ]
  
  def run(self, f, prox, x0, iteration, params, save_path, log_interval=-1):
    self.tk = params["alpha"]
    return super().run(f, prox, x0, iteration, params, save_path, log_interval)
  
  def __run_init__(self,f, prox,x0,iteration,params):
    self.k = 0
    self.xk1 = x0.copy()
    return super().__run_init__(f,prox,x0,iteration,params)

  def __iter_per__(self):
    self.k+=1
    beta = self.params["beta"]
    eps = self.params["eps"]
    restart = self.params["restart"]
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

# second order method      
class NewtonMethod(optimization_solver):
  def __init__(self,  dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = [
      "alpha",
      "beta",
      "eps",
      "backward"
    ]

  def __iter_per__(self):
    grad = self.__first_order_oracle__(self.xk)
    if self.check_norm(grad,self.params["eps"]):
      self.finish = True
      return
    H = self.__second_order_oracle__(self.xk)
    dk = self.__direction__(grad=grad,hess=H)
    lr = self.__step_size__(grad=grad,dk=dk)
    self.__update__(lr*dk)
        
  def __direction__(self, grad,hess):
    return - jnp.linalg.solve(hess,grad)
    
  def __step_size__(self, grad,dk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return line_search(self.xk,self.f,grad,dk,alpha,beta)

class SubspaceNewton(SubspaceGD):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key =["dim",
                      "reduced_dim",
                      "mode",
                      "backward",
                      "alpha",
                      "beta",
                      "eps"]

  def __iter_per__(self):
    reduced_dim = self.params["reduced_dim"]
    dim = self.params["dim"]
    mode = self.params["mode"]
    Mk = self.generate_matrix(dim,reduced_dim,mode)
    grad = self.subspace_first_order_oracle(self.xk,Mk)
    if self.check_norm(grad,self.params["eps"]):
      self.finish = True
      return
    H = self.subspace_second_order_oracle(self.xk,Mk)
    dk = self.__direction__(grad=grad,hess=H)
    lr = self.__step_size__(grad=grad,dk=dk,Mk=Mk)
    self.__update__(lr*Mk.T@dk)
  
  def __direction__(self, grad,hess):
    return - jnp.linalg.solve(hess,grad)
    
  def __step_size__(self, grad,dk,Mk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)

  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/(reduced_dim**0.5)
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")

class LimitedMemoryNewton(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.Pk = None
    self.index = 0
    self.params_key = [
      "reduced_dim",
      "threshold_eigenvalue",
      "alpha",
      "beta",
      "backward",
      "mode",
      "eps"
    ]
    
  def generate_matrix(self,matrix_size,gk,mode):
    # P^\top = [x_0,\nabla f(x_0),...,x_k,\nabla f(x_k)]
    dim = self.xk.shape[0]
    if mode == LEESELECTION:
      if self.Pk is None:
        self.Pk = jnp.zeros((matrix_size,dim),dtype = self.dtype)
      self.Pk = self.Pk.at[self.index].set(self.xk)
      self.Pk = self.Pk.at[self.index+1].set(gk)
      self.index+=2
      self.index %= matrix_size
    elif mode == RANDOM:
      self.Pk = jax_randn(matrix_size,dim,dtype = self.dtype)/matrix_size**0.5
    else:
      raise ValueError(f"{mode} is not implemented.")
    
  def subspace_second_order_oracle(self,x,Mk,threshold_eigenvalue):
    matrix_size = Mk.shape[0]
    H = super().subspace_second_order_oracle(x,Mk)
    sigma_m = get_minimum_eigenvalue(H)
    if sigma_m < threshold_eigenvalue:
        return H + (threshold_eigenvalue - sigma_m)*jnp.eye(matrix_size,dtype = self.dtype)
    else:
        return H                                                                                                                                                                                                                                                                                        
  
  def __iter_per__(self):
    matrix_size = self.params["reduced_dim"]
    threshold_eigenvalue = self.params["threshold_eigenvalue"]
    mode = self.params["mode"]
    gk = self.__first_order_oracle__(self.xk)
    self.generate_matrix(matrix_size,gk,mode)
    proj_gk = self.Pk@gk
    if self.check_norm(gk,self.params["eps"]):
      self.finish = True
      return
    Hk = self.subspace_second_order_oracle(self.xk,self.Pk,threshold_eigenvalue)
    dk = self.__direction__(grad=proj_gk,hess = Hk)
    lr = self.__step_size__(grad=proj_gk,dk=dk,Mk = self.Pk)
    self.__update__(lr*self.Pk.T@dk)
  
  def __direction__(self, grad,hess):
    return - jnp.linalg.solve(hess,grad)
  
  def __step_size__(self, grad,dk,Mk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)

class RandomizedBFGS(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.Bk_inv = None
    self.params_key = [
      "reduced_dim",
      "dim",
      "backward",
      "eps"
    ]
  
  def run(self, f, x0, iteration, params, save_path, log_interval=-1):
    self.__run_init__(f,x0,iteration,params)
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
      G = self.__first_order_oracle__(self.xk)
      self.update_save_values(i+1,time = T,func_values = F,grad_norm = G)
      if (i+1)%log_interval == 0 & log_interval != -1:
        logger.info(f'{i+1}: {self.save_values["func_values"][i+1]}')
        self.save_results(save_path)
    return
  
  def __run_init__(self, f, x0,iteration,params):
    self.Bk_inv = jnp.eye(x0.shape[0],dtype = self.dtype)
    return super().__run_init__(f, x0, iteration,params)
  
  def __iter_per__(self):
    reduced_dim = self.params["reduced_dim"]
    dim = self.params["dim"]
    grad,loss_k = self.__first_order_oracle__(self.xk,output_loss=True)
    if self.check_norm(grad,self.params["eps"]):
      self.finish = True
      return
    Hk = self.__second_order_oracle__(self.xk)
    self.__update__(-self.Bk_inv@grad,loss_k)
    Sk = self.generate_matrix(reduced_dim=reduced_dim,dim=dim)
    self.update_rbfgs(Hk,Sk)
    
  def update_rbfgs(self,Hk,Sk):
    dim = Hk.shape[0]
    G = Sk@jnp.linalg.solve(Sk.T@Hk@Sk,Sk.T)
    J = jnp.eye(dim,dtype = self.dtype) - G@Hk
    self.Bk_inv = G - J@self.Bk_inv@J.T
  
  def __update__(self, d,loss_k):
    xk1 = self.xk + d
    loss_k1 = self.f(xk1)
    if loss_k1 < loss_k:
      self.xk = xk1
    return    

  
  def generate_matrix(self,reduced_dim,dim):
    return jax_randn(dim,reduced_dim,dtype=self.dtype)

class SubspaceRNM(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.params_key = [
      "reduced_dim",
      "gamma",
      "c1",
      "c2",
      "alpha",
      "beta",
      "eps",
      "backward"
    ]
  
  def __iter_per__(self):
    reduced_dim = self.params["reduced_dim"]
    dim = self.xk.shape[0]
    Pk = self.generate_matrix(dim,reduced_dim)
    subspece_H = self.subspace_second_order_oracle(self.xk,Pk)
    projected_grad = self.subspace_first_order_oracle(self.xk,Pk)
    if self.check_norm(projected_grad,self.params["eps"]):
      self.finish = True
      return
    l_min = get_minimum_eigenvalue(subspece_H)
    L = max(0,-l_min)
    Mk = subspece_H + self.params["c1"]*L*jnp.eye(reduced_dim,dtype = self.dtype) + self.params["c2"]*jnp.linalg.norm(projected_grad)**self.params["gamma"]*jnp.eye(reduced_dim,dtype = self.dtype)
    dk = self.__direction__(projected_grad,Mk)
    s = self.__step_size__(projected_grad=projected_grad,
                           Mk=Pk,
                           dk = dk)
    self.__update__(s*Pk.T@dk)

  def __direction__(self, projected_grad,subspace_H):
    return - jnp.linalg.solve(subspace_H,projected_grad)
   
  def __step_size__(self, projected_grad,dk,Mk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return subspace_line_search(xk = self.xk,
                                func = self.f,
                                projected_grad=projected_grad,
                                dk=dk,
                                Mk=Mk,
                                alpha=alpha,
                                beta=beta)

  def generate_matrix(self,dim,reduced_dim,mode = "random"):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/(reduced_dim**0.5)
  
