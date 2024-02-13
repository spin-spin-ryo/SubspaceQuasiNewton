from algorithms.descent_method import optimization_solver
import jax.numpy as jnp
from utils.calculate import subspace_line_search,jax_randn,clipping_eigenvalues
import time
import numpy as np

class SubspaceQNM(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.Pk = None
    self.Hk = None
    self.projected_gradk = None
    self.index = 0
    self.params_key = [
      "alpha",
      "beta",
      "reduced_dim",
      "matrix_size",
      "dim",
      "backward",
      "lower_eigenvalue",
      "upper_eigenvalue",
      "eps",
      "mode"
    ]
  
  def __run_init__(self, f, x0, iteration,params):
    super().__run_init__(f, x0, iteration,params)
    dim = params["dim"]
    matrix_size = params["matrix_size"]
    reduced_dim = params["reduced_dim"]
    mode = params["mode"]
    self.Hk = jnp.eye(matrix_size,dtype = self.dtype)
    Q = self.generate_matrix(dim=dim,
                             reduced_dim=reduced_dim,
                             mode = "random")
    random_projected_grad = self.subspace_first_order_oracle(x = self.xk,Mk = Q)
    self.check_norm(random_projected_grad,params["eps"])
    self.update_Pk(matrix_size,
                   random_projected_grad,
                   Q,
                   mode)
    self.projected_gradk = self.subspace_first_order_oracle(self.xk,self.Pk)

  def __direction__(self, projected_grad):
    return -self.Hk@projected_grad
  
  def __step_size__(self, projected_grad,dk,Mk):
    alpha = self.params["alpha"]
    beta = self.params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=projected_grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)
  

  def __iter_per__(self):
    reduced_dim = self.params["reduced_dim"]
    dim = self.params["dim"]
    lower_eigenvalue = self.params["lower_eigenvalue"]
    upper_eigenvalue = self.params["upper_eigenvalue"]
    matrix_size = self.params["matrix_size"]
    
    dk = self.__direction__(self.projected_gradk)
    s = self.__step_size__(projected_grad=self.projected_gradk,
                           dk=dk,
                           Mk = self.Pk)
    self.__update__(s*self.Pk.T@dk)
    projected_gradk1 = self.subspace_first_order_oracle(self.xk,self.Pk)
    self.update_BFGS(sk = s*dk,yk = projected_gradk1 - self.projected_gradk)
    self.Hk = clipping_eigenvalues(self.Hk,lower=lower_eigenvalue,upper=upper_eigenvalue)
    Q = self.generate_matrix(dim=dim,
                             reduced_dim=reduced_dim,
                             mode = "random")
    random_projected_grad = self.subspace_first_order_oracle(self.xk,Q)
    if self.check_norm(random_projected_grad,self.params["eps"]):
      self.finish = True
      return
    self.update_Pk(matrix_size,
                   random_projected_grad,
                   Q)
    self.projected_gradk = self.subspace_first_order_oracle(self.xk,self.Pk)

  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)/(reduced_dim**0.5)
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")
  
  def update_BFGS(self,sk,yk):
    a = sk@yk
    if a < 1e-14:
      self.Hk = jnp.eye(sk.shape[0],dtype = self.dtype) 
      return 
    B = jnp.dot(jnp.expand_dims(self.Hk@yk,1),jnp.expand_dims(sk,0))
    S = jnp.dot(jnp.expand_dims(sk,1),jnp.expand_dims(sk,0))
    self.Hk = self.Hk + (a + self.Hk@yk@yk)*S/(a**2) - (B + B.T)/a
  
  def update_Pk(self,matrix_size,random_projected_grad,Qk,mode = "Identity"):
    dim = Qk.shape[1]
    # P^\top = [x_0/||x_0||,QTQ\nabla f(x_0)/||QTQ\nabla f(x_0)||,...,x_k/||x_k||,QTQ\nabla f(x_k)/||QTQ\nabla f(x_k)||]
    random_projected_grad_fullsize = Qk.T@random_projected_grad
    if jnp.linalg.norm(self.xk) < 1e-12:
      vector1 = jnp.zeros(dim,dtype=self.dtype)
    else:
      vector1 = self.xk/jnp.linalg.norm(self.xk)
    
    if jnp.linalg.norm(random_projected_grad_fullsize) < 1e-12:
      vector2 = jnp.zeros(dim,dtype=self.dtype)
    else:
      vector2 = random_projected_grad_fullsize/jnp.linalg.norm(random_projected_grad_fullsize)
    if self.Pk is None:
      if mode == "Identity":
        self.Pk = jnp.eye(matrix_size,dim,dtype = self.dtype)
      elif mode == "random":
        self.Pk = jax_randn(matrix_size,dim,dtype=self.dtype)/(dim**0.5)
      self.Pk = self.Pk.at[matrix_size-2].set(vector1)
      self.Pk = self.Pk.at[matrix_size-1].set(vector2)
      
    else:
      self.Pk = self.Pk.at[self.index].set(vector1)
      self.Pk = self.Pk.at[self.index+1].set(vector2)
      self.index += 2
      self.index %= matrix_size
      
    