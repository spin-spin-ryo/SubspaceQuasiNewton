# Subspace quasi newton method

## How to run numerical experiments
1. Select optimization problem and an algorithm and set the parameters in `make_config.py`.
2. Make config.json file using command `python make_config.py`.
3. Run numerical experiments using command `python main.py path/to/config.json`.
4. Check the results in `results/problem_name/problem_parameters/constraints_name/constraints_parameters/algorithm_name/algorithm_parameters` directory.
5. You can compare results using `python result_show.py`. with GUI interface.

## problems

### QUADRATIC
$$\min_{x\in \mathbb{R}^n} \frac{1}{2}x^\top A x + b^\top x$$
dim: n, if convex is True, then A become a semi definite matrix. data_name: only "random".

### SPARSEQUADRATIC
not used

### MATRIXFACTORIZATION
$$\min_{U,V} \|UV - X\|_F^2$$
data_name: only "movie", rank: the number of row of $U$, and column of $V$.

### MATRIXFACTORIZATION_COMPLETION
$$\min_{U,V} \|\mathcal{P}_{\Omega}(UV) - \mathcal{P}_{\Omega}(X)\|_F^2$$
data_name: only "movie", rank: the number of row of $U$, and column of $V$.

### LEASTSQUARE
not used

### MLPNET
linear neural network:
$$\min_{w} \sum_{i=1}^m \mathcal{L}(\mathcal{W}(w,x_i),y_i)$$
$(x_i,y_i)$:dataset, layers_size: [(in_features,out_feafures,use bias or not),], activation: activation function name (see `utils/select.py`), criterion: type of loss function (only 'CrossEntropy')

### CNN
convolutional neural network:
$$\min_{w} \sum_{i=1}^m \mathcal{L}(\mathcal{W}(w,x_i),y_i)$$
$(x_i,y_i)$:dataset,
layers_size: [(input_channels,output_channelskernel_size,bias_flag)],
activation: activation function name (see `utils/select.py`),
criterion: type of loss function (only 'CrossEntropy')

### SOFTMAX
minimizing softmax loss function.</br>
data_name: "Scotus" or "news20"

### LOGISTIC
minimizing logistic loss function</br>
data_name:"rcv1" or "news20" or "random".

### REGULARIZED
set `problem_name = REGULARIZED + other_problem_name`.
minimizing regularized function 
$$\min_x f(x) + \lambda \|x\|_p^p$$
coeff: $\lambda$,
ord: $p$,
Fused: only False

## constraints

### POLYTOPE
$${ x| Ax-b \le 0}$$
data_name:only "random",
dim: the dimension of $x$,
constraints_num: the dimension of $b$

### NONNEGATIVE
$${x | x_i \ge 0}$$
dim: the dimension of $x$

### QUADRATIC
$${x| \frac{1}{2}x^\top A_i x + b_i^\top x + c_i \le 0, i = 1,...,m}$$
data_name: only "random",
dim: the dimension of $x$,
constraints_num: m

### FUSEDLASSO
$${x| \|x\|_1 \le s_1, \sum_{i} |x_{i+1} - x_i|\le s_2}$$
threshold1: $s_1$,
threshold2: $s_2$

### BALL
$${x| \|x\|_p^p \le s}$$
ord: $p$,
threshold: $s$

## algorithms
#### backward parameters
 True: use automatic differentiation (very fast, but memory leak sometimes happens (CNN))
  DD: use directional derivative with automatic differentiation (efficiency depends on the dimension, no error)
  FD: use finite difference (efficiency depends on the dimension, error exists)

### GD(Gradient descent)
lr: step size,
eps: stop criteria,
linesearch: if true, use armijo line search with $\alpha = 0.3, \beta = 0.8$.

### SGD(Subspace gradient descent[https://arxiv.org/abs/2003.02684])
lr: step size,
eps: stop criteria,
reduced_dim: size of random matrix,
mode: only "random",
linesearch: if true, use armijo line search with $\alpha = 0.3, \beta = 0.8$.

### AGD(Accelerated Gradient descent)
lr: step size,
eps: stop criteria,
restart: if true, use function value restart.

### BFGS
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria

### LimitedMemoryBFGS[https://en.wikipedia.org/wiki/Limited-memory_BFGS]
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria
memory_size: the number of past data.

### BacktrackingProximalGD(proximal gradient descent with line search)
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria

### BacktrackingAcceleratedProximalGD(accelerated proximal gradient descent with line search)
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria,
restart: if true, use function value restart.

### Newton method
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria,

### SubspaceNewton[https://arxiv.org/abs/1905.10874]
dim: the dimension of problem,
reduced_dim, the size of random matrix,
mode: the type of random matrix(only "random"),
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria,

### LimitedMemoryNewton[https://link.springer.com/article/10.1007/s12532-022-00219-z]
reduced_dim: the size of subspace matrix,
threshold_eigenvalue: parameter of clipping eigenvalues,
mode: the type of random matrix(only "LEESELECTION"),
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria,

### SubspaceRNM(subspace regularized newton method[https://arxiv.org/abs/2209.04170])
reduced_dim:the size of random matrix,
please refer to the paper for other parameters.


### Proposed method
alpha: parameter of line search,
beta: parameter of line search,
eps: stop criteria,
reduced_dim: the size of random matrix,
matrix_size: the size of subspace matrix(not random),
dim: the dimension of problem,
lower_eigenvalue: the parameter of clipping eigenvalues,
upper_eigenvalue: the parameter of clipping eigenvalues,
mode: the type of random matrix(only "random")