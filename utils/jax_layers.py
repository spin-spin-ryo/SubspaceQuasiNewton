import jax.numpy as jnp
from jax.lax import conv_general_dilated,reshape,transpose
from flax.linen import avg_pool
import jax.nn as nn
from jax import jit,vmap
from operator import getitem

def linear(input,weight,bias = None):
  # input: (data_size, input_size)
  # weight:(output_size,input_size)
  if bias is None:
    return jnp.dot(input,weight.T)
  else:
    return jnp.dot(input,weight.T) + bias

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  if bias is None:
    return conv_general_dilated(lhs = input, rhs=weight, window_strides=(stride, stride), padding=[(padding,padding),(padding,padding)], lhs_dilation=(dilation, dilation), rhs_dilation=(dilation, dilation), feature_group_count=groups, batch_group_count=groups)
  else:
    return conv_general_dilated(lhs = input, rhs=weight, window_strides=(stride, stride), padding=[(padding,padding),(padding,padding)], lhs_dilation=(dilation, dilation), rhs_dilation=(dilation, dilation), feature_group_count=groups, batch_group_count=groups) + reshape(bias,(1,bias.size,1,1))

def avg_pool2d(input, kernel_size , stride=None, padding=0):
  if stride is None:
    strides = (kernel_size,kernel_size)
  else:
    strides = (stride,stride)
  return transpose(avg_pool(inputs = transpose(input,(0,2,3,1)), window_shape = (kernel_size,kernel_size), strides=strides, padding=[(padding,padding),(padding,padding)]),(0,3,1,2))

@jit
def cross_entropy_loss(logits, labels):
    logits = nn.log_softmax(logits)
    loss = vmap(getitem)(logits, labels.astype(jnp.int64))
    loss = -loss.mean()
    return loss