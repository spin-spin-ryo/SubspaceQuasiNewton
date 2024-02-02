import jax.nn as nn
from utils.jax_layers import cross_entropy_loss

def get_activation(activation_name):
  if activation_name == "sigmoid":
    activation = nn.sigmoid
  elif activation_name == "relu":
    activation = nn.relu
  else:
    raise ValueError("No activation")
  return activation

def get_criterion(criterion_name):
  if criterion_name == "CrossEntropy":
    criterion = cross_entropy_loss
  else:
    raise ValueError("No criterion")
  return criterion
    