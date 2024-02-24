from problems.objectives import *
from problems.constraints import *
from environments import *
import pickle
from utils.calculate import generate_symmetric,generate_semidefinite,nonnegative_projection,L1projection,BallProjection,BoxProjection,jax_randn
from utils.select import get_activation,get_criterion
import os
from jax.lax import transpose
import jax.nn as nn
from sklearn.datasets import load_svmlight_file
from jax.experimental.sparse import BCSR
        

objective_properties_key ={
    QUADRATIC:["dim","convex","data_name"],
    SPARSEQUADRATIC:["dim","data_name"],
    MATRIXFACTORIZATION:["data_name","rank"],
    MATRIXFACTORIZATION_COMPLETION:["data_name","rank"],
    LEASTSQUARE:["data_name","data_size","dim"],
    MLPNET: ["data_name","layers_size","activation","criterion"],
    CNN: ["data_name","layers_size","activation","criterion"],
    SOFTMAX:["data_name"],
    LOGISTIC:["data_name"],
    SPARSEGAUSSIANPROCESS:["data_name","reduced_data_size","kernel_mode"],
    REGULARIZED: ["coeff","ord","Fused"]
}

constraints_properties_key = {
    POLYTOPE:["data_name","dim","constraints_num"],
    NONNEGATIVE:["dim"],
    QUADRATIC:["data_name","dim","constraints_num"],
    FUSEDLASSO: ["threshold1","threshold2"],
    BALL:["ord","threshold"],
    HUBER:["delta","threshold"]
}

def generate_objective(function_name,function_properties):
    use_regularized = REGULARIZED in function_name
    if use_regularized:
        function_name = function_name.replace(REGULARIZED,"")
    if function_name == QUADRATIC:
        f = generate_quadratic(function_properties)
    elif function_name == SPARSEQUADRATIC:
        f = generate_sparse_quadratic(function_properties)
    elif function_name == MATRIXFACTORIZATION:
        f = generate_matrix_factorization(function_properties)
    elif function_name == MATRIXFACTORIZATION_COMPLETION:
        f = generate_matrix_factorization_completion(function_properties)
    elif function_name == LEASTSQUARE:
        f = generate_least_square(function_properties)
    elif function_name == MLPNET:
        f = generate_mlpnet(function_properties)
    elif function_name == CNN:
        f = generate_cnn(function_properties)
    elif function_name == SOFTMAX:
        f = generate_softmax(function_properties)
    elif function_name == LOGISTIC:
        f = generate_logistic(function_properties)
    elif function_name == SPARSEGAUSSIANPROCESS:
        f = generate_sparse_gaussian_process(function_properties)
    else:
        raise ValueError(f"{function_name} is not implemented.")
    
    if use_regularized:
        f = wrapping_regularization(f,function_properties)
    return f

def generate_constraints(constraints_name, constraints_properties):
    prox = None
    if constraints_name == POLYTOPE:
        constraints = generate_polytope(constraints_properties)
    elif constraints_name == NONNEGATIVE:
        prox = nonnegative_projection
        constraints = generate_nonnegative(constraints_properties)
    elif constraints_name == QUADRATIC:
        constraints = generate_quadratic_constraints(constraints_properties)
    elif constraints_name == FUSEDLASSO:
        constraints = generate_fusedlasso(constraints_properties)
    elif constraints_name == BALL:
        ord = constraints_properties["ord"]
        threshold = constraints_properties["threshold"]
        if ord == 1:
            def prox(x,t):
                return L1projection(x,radius=threshold)
        elif ord == 2:
            def prox(x,t):
                return BallProjection(x,radius=threshold**(0.5))
        else:
            raise ValueError("No proximal function")
        constraints = generate_ball(constraints_properties)
    elif constraints_name == HUBER:
        constraints = generate_huber(constraints_properties)
    elif constraints_name == NOCONSTRAINTS:
        constraints = None
    else:
        raise ValueError(f"{constraints_name} is not implemented.")
    return constraints, prox

def generate_initial_points(func,function_name,constraints_name,function_properties):
    dim = func.get_dimension()
    function_name = function_name.replace(REGULARIZED,"")
    # 非負制約の時のみすべて1
    if constraints_name == NONNEGATIVE:
        x0 = jnp.ones(dim)
        return x0
    
    if function_name == MLPNET:
        if function_properties["data_name"] == "mnist":
            x0 = jnp.load(os.path.join(DATAPATH,"mnist","mlpnet",f"init_param_{dim}.npy"))
            return x0
    
    if function_name == CNN:
        if function_properties["data_name"] == "mnist":
            # dim:33738
            x0 = jnp.load(os.path.join(DATAPATH,"mnist","cnn","init_param.npy"))
            return x0
    if function_name == MATRIXFACTORIZATION_COMPLETION or function_name == MATRIXFACTORIZATION:
        x0 = jnp.ones(dim)
        return x0
    
    x0 = jnp.zeros(dim)
    return x0

def wrapping_regularization(f,properties):
    ord = int(properties["ord"])
    coefficient = float(properties["coeff"])
    use_fused = bool(properties["Fused"])
    if use_fused:
        A = None
        pass
    else:
        A = None
    params = [ord,coefficient,A]
    return regularzed_wrapper(f,params)

def generate_quadratic(properties):
    dim = int(properties["dim"])
    convex = properties["convex"]
    data_name = properties["data_name"]
    if data_name == "random":
        if convex:
            # rankは適当
            rank = dim//2
            Quadratic_data_path = os.path.join(DATAPATH,"quadratic","convex")
            filename_Q = f"Q_{dim}_{rank}.npy"
            filename_b = f"b_{dim}_{rank}.npy"
            if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
                Q = jnp.load(os.path.join(Quadratic_data_path,filename_Q))
                b = jnp.load(os.path.join(Quadratic_data_path,filename_b))
                c = 0
            else:
                Q = generate_semidefinite(dim=dim,rank=rank)
                b = jax_randn(dim)
                c = 0
                os.makedirs(Quadratic_data_path,exist_ok=True)
                jnp.save(os.path.join(Quadratic_data_path,filename_Q),Q)
                jnp.save(os.path.join(Quadratic_data_path,filename_b),b)
        else:   
            Quadratic_data_path = os.path.join(DATAPATH,"quadratic","nonconvex")
            filename_Q = f"Q_{dim}.npy"
            filename_b = f"b_{dim}.npy"
            if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
                Q = jnp.load(os.path.join(Quadratic_data_path,filename_Q))
                b = jnp.load(os.path.join(Quadratic_data_path,filename_b))
                c = 0
            else:
                Q = generate_symmetric(dim=dim)
                b = jax_randn(dim)
                c = 0
                os.makedirs(Quadratic_data_path,exist_ok=True)
                jnp.save(os.path.join(Quadratic_data_path,filename_Q),Q)
                jnp.save(os.path.join(Quadratic_data_path,filename_b),b)                    
    params = [Q,b,c]
    f = QuadraticFunction(params=params)
    return f

def generate_sparse_quadratic(properties):
    dim = int(properties["dim"])
    data_name = properties["data_name"]
    if data_name == "random":
        Quadratic_data_path = os.path.join(DATAPATH,"sparse_quadratic")
        filename_Q = f"Q_{dim}.npy"
        filename_b = f"b_{dim}.npy"
        if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
            Q = jnp.load(os.path.join(Quadratic_data_path,filename_Q))
            b = jnp.load(os.path.join(Quadratic_data_path,filename_b))
            c = 0
        else:
            Q = jax_randn(dim)
            b = jax_randn(dim)
            c = 0
            os.makedirs(Quadratic_data_path,exist_ok=True)
            jnp.save(os.path.join(Quadratic_data_path,filename_Q),Q)
            jnp.save(os.path.join(Quadratic_data_path,filename_b),b)    
    params = [Q,b,c]
    f = SparseQuadraticFunction(params=params)
    return f

def generate_matrix_factorization(properties):
    data_name = properties["data_name"]
    rank = int(properties["rank"])
    if data_name == "movie":
        U = jnp.load(os.path.join(DATAPATH,"movie","movie_train_100k.npy"))
    
    params = [U,rank]
    f = MatrixFactorization_2(params=params)
    return f

def generate_matrix_factorization_completion(properties):
    data_name = properties["data_name"]
    rank = int(properties["rank"])
    if data_name == "movie":
        U = jnp.load(os.path.join(DATAPATH,"movie","movie_train_100k.npy"))
    
    params = [U,rank]
    f = MatrixFactorization_Completion(params=params)
    return f

def generate_least_square(properties):
    data_name = properties["data_name"]
    data_size = int(properties("data_size"))
    dim = int(properties("dim"))
    if data_name == "random":
        data_path = os.path.join(DATAPATH,"least_square")
        filename_A = f"A_{dim}_{data_size}.npy"
        filename_b = f"b_{dim}_{data_size}.npy"
        if os.path.exists(os.path.join(data_path,filename_A)):
            A = jnp.load(os.path.join(data_path,filename_A))
            b = jnp.load(os.path.join(data_path,filename_b))
        else:
            A = jax_randn(data_size,dim)
            b = jax_randn(data_size)
            os.makedirs(data_path,exist_ok=True)
            jnp.save(os.path.join(data_path,filename_A),A)
            jnp.save(os.path.join(data_path,filename_b),b)
    
    params = [A,b]
    f = LeastSquare(params)
    return f

def generate_mlpnet(properties):
    data_name = properties["data_name"]
    layers_size = properties["layers_size"]
    activation_name = properties["activation"]
    criterion_name = properties["criterion"]
    activation = get_activation(activation_name)
    criterion = get_criterion(criterion_name)
    
    if data_name == "mnist":
        data_path = os.path.join(DATAPATH,"mnist","mlpnet")
        data = jnp.load(os.path.join(data_path,"mnist_data.npy"))
        label = jnp.load(os.path.join(data_path,"mnist_label.npy"))
    
    else:
        raise ValueError(f"{data_name} does not exist.")
    params = [data,label,layers_size]
    f = MLPNet(params,activation=activation,criterion=criterion)
    return f

def generate_cnn(properties):
    data_name = properties["data_name"]
    layers_size = properties["layers_size"]
    activation_name = properties["activation"]
    criterion_name = properties["criterion"]
    activation = get_activation(activation_name)
    criterion = get_criterion(criterion_name)

    if data_name == "mnist":
        data_num = 30000
        data_path = os.path.join(DATAPATH,"mnist","cnn")
        data = jnp.load(os.path.join(data_path,"images.npy"))
        label = jnp.load(os.path.join(data_path,"labels.npy"))
        data_size = data.shape[2]
        print("data_size:",data_size)
        class_num = (jnp.unique(label)).shape[0]
    
    params = [data[:data_num],label[:data_num],class_num,data_size,layers_size]
    f = CNNet(params,criterion=criterion,activation=activation)
    return f

def generate_softmax(properties):
    data_name = properties["data_name"]
    if data_name == "Scotus":
        path_dataset = os.path.join(DATAPATH,"classification","scotus_lexglue_tfidf_train.svm.bz2")
        X,y = load_svmlight_file(path_dataset)
        num_classes = len(jnp.unique(y))
        y=nn.one_hot(y,num_classes=num_classes)
        
    elif data_name == "news20":
        path_dataset = os.path.join(DATAPATH,"classification","news20.bz2")
        X,y = load_svmlight_file(path_dataset)
        num_classes = len(jnp.unique(y))
        y=nn.one_hot(y,num_classes=num_classes)
    X = BCSR.from_scipy_sparse(X)
    y = jnp.array(y)
    params = [X,y]
    f = softmax(params)
    return f

def generate_logistic(properties):
    data_name = properties["data_name"]
    if data_name == "rcv1":
        # [20242,47236]
        path_dataset = os.path.join(DATAPATH,"classification","rcv1_train.binary.bz2")
        X,y = load_svmlight_file(path_dataset)
        
    elif data_name == "news20":
        # [19996,1355191]
        path_dataset = os.path.join(DATAPATH,"classification","news20.binary.bz2")
        X,y = load_svmlight_file(path_dataset)
    X = BCSR.from_scipy_sparse(X)
    y = jnp.array(y)
    
    params = [X,y]
    f = logistic(params)
    return f

def generate_sparse_gaussian_process(properties):
    data_name = properties["data_name"]
    reduced_data_size = int(properties["reduced_data_size"])
    kernel_mode = properties["kernel_mode"]
    if data_name == "E2006":
        path_dataset = os.path.join(DATAPATH,"regression","E2006.train.bz2")
        X,y = load_svmlight_file(path_dataset)
        data_dim = X.shape[1]
        X = BCSR.from_scipy_sparse(X)
        y = jnp.array(y)
    elif data_name == "ackley":
        X = jnp.load(os.path.join(DATAPATH,"regression","ackley_X.npy"))
        y = jnp.load(os.path.join(DATAPATH,"regression","ackley_y.npy"))
        data_dim = 10000
    params = [X,y,data_dim,reduced_data_size,kernel_mode]
    f = SparseGaussianProcess(params)
    return f


def generate_polytope(properties):
    data_name = properties["data_name"]
    dim = properties["dim"]
    constraints_num = properties["constraints_num"]
    if data_name == "random":
        data_path = os.path.join(DATAPATH,"polytope")
        filename_A = f"A_{dim}_{constraints_num}.npy"
        b = jnp.ones(constraints_num)
        if os.path.exists(os.path.join(data_path,filename_A)):
            A = jnp.load(os.path.join(data_path,filename_A))
        else:
            A = jax_randn(constraints_num,dim)
            os.makedirs(data_path,exist_ok=True)
            jnp.save(os.path.join(data_path,filename_A),A)
    
    params = [A,b]
    con= Polytope(params)
    return con

def generate_nonnegative(properties):
    dim = properties["dim"]
    con = NonNegative(params=[dim])
    return con

def generate_quadratic_constraints(properties):
    data_name = properties["data_name"]
    dim = properties["dim"]
    constraints_num = properties["constraints_num"]
    if data_name == "random":
        rank = dim//2
        data_path = os.path.join(DATAPATH,"quadratic_constraints")
        filename_Q = f"Q_{dim}_{rank}_{constraints_num}.npy"
        filename_b = f"b_{dim}_{rank}_{constraints_num}.npy"
        if os.path.exists(os.path.join(data_path,filename_Q)):
            Q = jnp.load(os.path.join(data_path,filename_Q))
            b = jnp.load(os.path.join(data_path,filename_b))
            c = -jnp.ones(constraints_num)
        else:
            P = jax_randn(constraints_num,dim,dim)
            Q = P@transpose(P,(0,2,1))/dim
            b = jax_randn(constraints_num,dim)
            c = -jnp.ones(constraints_num)
            os.makedirs(data_path,exist_ok=True)
            jnp.save(os.path.join(data_path,filename_Q),Q)
            jnp.save(os.path.join(data_path,filename_b),b)
    params = [Q,b,c]
    con = Quadratic(params)
    return con

def generate_fusedlasso(properties):
    a = float(properties["threshold1"])
    b = float(properties["threshold2"])
    params = [a,b]
    con = Fused_Lasso(params)
    return con

def generate_ball(properties):
    ord = properties["ord"]
    threshold = properties["threshold"]
    params = [threshold,ord]
    con = Ball(params)
    return con

def generate_huber(properties):
    delta = properties["delta"]
    threshold = properties["threshold"]
    params = [delta,threshold]
    con = Huber(params)
    return con

