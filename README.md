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
$$\{ x| Ax-b \le 0\}$$
data_name:only "random",
dim: the dimension of $x$,
constraints_num: the dimension of $b$

### NONNEGATIVE
$$\{x | x_i \ge 0\}$$
dim: the dimension of $x$

### QUADRATIC
$$\{x| \frac{1}{2}x^\top A_i x + b_i^\top x + c_i \le 0, i = 1,...,m\}$$
data_name: only "random",
dim: the dimension of $x$,
constraints_num: m

### FUSEDLASSO
$$\{x| \|x\|_1 \le s_1, \sum_{i} |x_{i+1} - x_i|\le s_2\}$$
threshold1: $s_1$,
threshold2: $s_2$

### BALL
$$\{x| \|x\|_p^p \le s\}$$
ord: $p$,
threshold: $s$

