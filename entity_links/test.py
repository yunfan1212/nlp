

import tqdm
from neurai.config import set_platform
set_platform(platform='cpu')
from neurai.nn import Module, Linear, Relu
from neurai import datasets
from neurai.nn.layer.loss import softmax_cross_entropy
import jax.numpy as nnp
import jax
from jax import jit
import optax as opt
from neurai.nn import sigmoid_binary_cross_entropy

import jax





