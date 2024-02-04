from jax import jacrev,jacfwd
import jax.numpy as jnp
from jax.lax import transpose
from jax import random,jvp
from environments import key
import numpy as np

def hessian(f):
    return jacfwd(jacrev(f))


def inverse_xy(x,y):
    dim = x.shape[0]
    dtype = x.dtype
    # (I+xy^\top)^{-1}を求める.
    return jnp.eye(dim,dtype=dtype) - (jnp.expand_dims(x,1)@jnp.expand_dims(y,0))/(1+x@y)

def get_minimum_eigenvalue(H):
    return jnp.min(jnp.linalg.eigvals(H))

def get_maximum_eigenvalue(H):
    return jnp.max(jnp.linalg.eigvals(H))

def line_search(xk,func,grad,dk,alpha,beta,loss = None):
  if loss is None:
    loss = func(xk)
  lr = 1
  while loss - func(xk + lr*dk) < -alpha*lr*grad@dk:
    lr *= beta 
  return lr

def subspace_line_search(xk,func,projected_grad,dk,Mk,alpha,beta,loss = None):
    if loss is None:
      loss = func(xk)
    lr = 1
    proj_dk = transpose(Mk,(1,0))@dk
    while loss.item() - func(xk + lr*proj_dk) < -alpha*lr*projected_grad@dk:
      lr *= beta 
    return lr

def generate_semidefinite(dim,rank):
   global key
   P = random.normal(key,(dim,rank))
   key, _ = random.split(key)
   return P@transpose(P,(1,0))/dim

def generate_symmetric(dim,device):
   global key
   P = random.normal(key,(dim,dim))
   key, _ = random.split(key)
   return (P + transpose(P,(1,0)))/2

def jax_randn(*args,dtype = jnp.float32):
   global key
   P = random.normal(key,args).astype(dtype)
   key, _ = random.split(key)
   return P

def nonnegative_projection(x,t):
    y = np.zeros(x.size,dtype=x.dtype)
    index = x > 0
    y[index] = x[index] 
    return jnp.array(y)

def BallProjection(x,radius = 1):
    if x@x <= radius*radius:
        return x
    else:
        return (radius)*x/jnp.linalg.norm(x)

def BoxProjection(x,radius = 1):
    y = np.array(x.copy())
    y[y>radius] = radius
    y[y<-radius] = -radius
    return jnp.array(y)

def L1projection(x,radius = 1):
  if jnp.linalg.norm(x,ord=1)<=radius:
    return x
  else:
    x_ = x.copy()
    
    x_/=radius
    y = jnp.sort(jnp.abs(x_))[0]
    l = 0
    r = y.shape[0]
    while r-l > 1:
      m = int((l+r)/2)
      lam = y[m]
      z = y -lam
      index = z >0
      if jnp.sum(z[index])>1:
        l = m
      else:
        r = m
    lam = (jnp.sum(y[r:]) -1)/y[r:].shape[0]
    z = np.zeros(y.shape,dtype = y.dtype)
    z[x_>lam] = (x_-lam)[x_ >lam]
    z[x_<-lam] =(x_+lam)[x_ <-lam]
    return jnp.array(z*radius)

def get_jvp(func,x,M):
  reduced_dim = M.shape[0]
  d = np.zeros(reduced_dim,dtype = x.dtype)
  for i in range(reduced_dim):
    _,directional_derivative= jvp(func,(x,),(M[i],))
    d[i] = directional_derivative
  return jnp.array(d)