import jax.numpy as jnp
from jax.lax import conv_general_dilated,reshape,transpose
from flax.linen import avg_pool
import jax.nn as nn
from jax import jit,vmap,random
from operator import getitem
import math


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


def init_cnn_params(rng_key, params):
    """
    CNNetの初期値ベクトルを生成する。
    He初期化を使用。
    
    params: [X, y, class_num, data_size, layer_config]
    """
    _, _, class_num, data_size, layer_config = params
    used_variables = []

    key = rng_key
    for (in_ch, out_ch, kernel_size, bias_flag) in layer_config:
        # He初期化：N(0, sqrt(2/fan_in))
        fan_in = in_ch * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        key, subkey_w = random.split(key)
        W = std * random.normal(subkey_w, (out_ch, in_ch, kernel_size, kernel_size))
        used_variables.append(W.flatten())

        if bias_flag == 1:
            key, subkey_b = random.split(key)
            b = jnp.zeros((out_ch,))  # CNNでは通常0初期化
            used_variables.append(b.flatten())

        # 出力画像サイズ更新（padding=2, stride=2 pooling前提）
        data_size = data_size + 4 + 1 - kernel_size
        data_size //= 2

    # 全結合層
    dim = data_size * data_size * out_ch
    fan_in = dim
    fan_out = class_num
    std = math.sqrt(2.0 / fan_in)
    key, subkey_fc_w = random.split(key)
    W_fc = std * random.normal(subkey_fc_w, (class_num, dim))
    used_variables.append(W_fc.flatten())

    key, subkey_fc_b = random.split(key)
    b_fc = jnp.zeros((class_num,))
    used_variables.append(b_fc.flatten())

    # すべて連結
    init_vector = jnp.concatenate(used_variables)
    return init_vector

