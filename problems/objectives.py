#目的関数クラス
import jaxlib
import jax.numpy as jnp
from jax import jit
from functools import partial
import utils.jax_layers as F

class Objective:
  def __init__(self,params):
    self.params = params
    return
  
  def get_dimension(self):
    return

  @partial(jit, static_argnums=0)
  def __call__(self,x):
    return
  
  def set_type(self,dtype):
    for i in range(len(self.params)):
      if isinstance(self.params[i],jaxlib.xla_extension.ArrayImpl):
        self.params[i] = self.params[i].astype(dtype)
    return
  
  
class QuadraticFunction(Objective):
  # params: [Q,b,c]
  @partial(jit, static_argnums=0)
  def __call__(self,x):
    Q = self.params[0]
    b = self.params[1]
    c = self.params[2]
    return 1/2*Q@x@x+b@x + c
  
  def get_dimension(self):
    return self.params[0].shape[0]

class SparseQuadraticFunction(Objective):
  @partial(jit, static_argnums=0)
  def __call__(self,x):
    Q = self.params[0]
    b = self.params[1]
    c = self.params[2]
    return jnp.sum(Q*x*x) + b@x +c 
  
class MatrixFactorization_1(Objective):
  # ||UV - X||_F
  # params: [X, rank]
  
  def __init__(self, params):
    super().__init__(params)
    self.row = params[0].shape[1]
    self.column = params[0].shape[0]
    self.rank = params[1]
  
  def get_dimension(self):
    return self.rank*self.row + self.rank*self.column

  @partial(jit, static_argnums=0)
  def __call__(self,x):
    assert len(x)==self.rank*self.row + self.rank*self.column
    W = x[:self.rank*self.column].reshape(self.column,self.rank)
    V = x[self.rank*self.column:].reshape(self.rank,self.row)
    return jnp.linalg.norm(self.params[0]-W@V,ord = "fro")
  
class MatrixFactorization_2(Objective):
  # ||UV - X||_F^2
  # params: [X, rank]
  def __init__(self, params):
    super().__init__(params)
    self.row = params[0].shape[1]
    self.column = params[0].shape[0]
    self.rank = params[1]
  
  def get_dimension(self):
    return self.rank*self.row + self.rank*self.column
  @partial(jit, static_argnums=0)
  def __call__(self,x):
    assert len(x)==self.rank*self.row + self.rank*self.column
    W = x[:self.rank*self.column].reshape(self.column,self.rank)
    V = x[self.rank*self.column:].reshape(self.rank,self.row)
    return jnp.linalg.norm(self.params[0]-W@V,ord = "fro")**2

class MatrixFactorization_Completion(Objective):
  # ||P_\Omega(UV) - P_{\Omega}(X)||_F
  def __init__(self, params):
    super().__init__(params)
    self.row = params[0].shape[1]
    self.column = params[0].shape[0]
    self.index = self.params[0]>0
    self.rank = self.params[1]
  
  def get_dimension(self):
    return self.rank*self.row + self.rank*self.column

  @partial(jit, static_argnums=0)
  def __call__(self,x):
    assert len(x)==self.rank*self.row + self.rank*self.column
    W = x[:self.rank*self.column].reshape(self.column,self.rank)
    V = x[self.rank*self.column:].reshape(self.rank,self.row)
    return jnp.linalg.norm((self.params[0]-W@V)[self.index])**2

class LeastSquare(Objective):
  # ||Ax-b||^2
  # params: [A,b]
  @partial(jit, static_argnums=0)
  def __call__(self,x):
    return jnp.linalg.norm(self.params[1]-self.params[0]@x)**2
  
  def get_dimension(self):
    return self.params[0].shape[1]

class MLPNet(Objective):
  # classification
  def __init__(self, params,criterion,activation):
    # params: [X,y,[(input_size,output_size,bias_flag_i)]]
    super().__init__(params)
    self.layer_size = params[1]
    self.criterion = criterion
    self.activate = activation
  
  def get_dimension(self):
    used_variables_num = 0
    for input_size,output_size,bias_flag in self.params[2]:
      used_variables_num += input_size*output_size
      if bias_flag == 1:
        used_variables_num += output_size
    return used_variables_num

  @partial(jit, static_argnums=0)
  def __call__(self,x):
    W = []
    bias = []
    used_variables_num = 0
    for input_size,output_size,bias_flag in self.params[2]:
      w = x[used_variables_num:used_variables_num+input_size*output_size].reshape(output_size,input_size)
      used_variables_num += input_size*output_size
      if bias_flag == 1:
        b = x[used_variables_num:used_variables_num + output_size]
        used_variables_num += output_size
      else:
        b = None
      W.append(w)
      bias.append(b)
    
    X = self.params[0]
    for i in range(len(self.params[2])):
      if bias[i] is not None:
        X = self.activate(F.linear(X, W[i], bias=bias[i]))
      else:
        X = self.activate(F.linear(X, W[i]))
    return self.criterion(X, self.params[1])

class CNNet(Objective):
    # classification
    # params: [X,y,class_num,data_size,[(input_channels,output_channels,kernel_size,bias_flag_i)]]
    def __init__(self, params,criterion,activation):
      super().__init__(params)
      self.criterion = criterion
      self.activate = activation

    def get_dimension(self):
      used_variables_num = 0
      data_size = self.params[3]
      class_num = self.params[2]
      for input_channels,output_channels,kernel_size,bias_flag in self.params[4]:
        used_variables_num += input_channels*output_channels*kernel_size*kernel_size
        data_size = data_size + 4 + 1 - kernel_size
        data_size //= 2
        if bias_flag == 1:
          used_variables_num += output_channels
      used_variables_num += data_size*data_size*output_channels*class_num + class_num
      return used_variables_num

    @partial(jit, static_argnums=0)
    def __call__(self,x):
      used_variables_num = 0
      for input_channels,output_channels,kernel_size,bias_flag in self.params[4]:
        # (data_num,input_channnels,height,width) -> (data_num,output_channels,height + 2*padding + 1 - kernel_size, width + 2*padding + 1 - kernel_size,width)
        if bias_flag == 1:
          z = F.conv2d(input = z,
                       weight = x[used_variables_num:used_variables_num + input_channels*output_channels*kernel_size*kernel_size].reshape(output_channels,input_channels,kernel_size,kernel_size),
                      bias = x[used_variables_num + input_channels*output_channels*kernel_size*kernel_size:used_variables_num + input_channels*output_channels*kernel_size*kernel_size + output_channels],
                      padding = 2)
          used_variables_num += input_channels*output_channels*kernel_size*kernel_size
          used_variables_num += output_channels
        else:
          z = F.conv2d(input = z,
                       weight = x[used_variables_num:used_variables_num + input_channels*output_channels*kernel_size*kernel_size].reshape(output_channels,input_channels,kernel_size,kernel_size),
                       padding = 2)
          used_variables_num += input_channels*output_channels*kernel_size*kernel_size
        z = self.activate(z)
        # (data_num,input_channnels,height,width) -> (data_num,input_channnels,height//2,width//2)
        z = F.avg_pool2d(input = z , kernel_size= 2)
      z = z.reshape(z.shape[0], -1)
      dim = z.shape[1]
      class_num = self.params[2]
      
      W = x[used_variables_num:used_variables_num+dim*class_num].reshape(class_num,dim)
      used_variables_num += dim*class_num
      b = x[used_variables_num:]
      # print(z.shape,W.shape,b.shape)
      z = F.linear(z, W, bias=b)
      return self.criterion(z, self.params[1])

# class Styblinsky(Objective):
#   def __call__(self,x):
#     return 1/2*jnp.sum(x[:self.params[0]]**4 - 16*x[:self.params[0]]**2 + 5*x[:self.params[0]])

# class Ackley(Objective):
#   def __call__(self, x):
#     return 20 - 20*jnp.exp( -0.2 * jnp.sqrt(jnp.mean(x[:self.params[0]]**2))) + jnp.exp(torch.tensor(1,dtype = x.dtype,device = x.device)) - torch.exp(torch.mean(torch.cos(2 * 3.141592653589793238 * x[:self.params[0]])))

# class Rastrigin(Objective):
#   def __call__(self,x):
#     return 10*x.shape[0] + torch.sum(x**2) -10*torch.sum(torch.cos(2 * 3.141592653589793238 * x))
  
# class Schwefel(Objective):
#   def __call__(self,x):
#     return - torch.sum(x*torch.sin(torch.sqrt(torch.abs(x))))
    