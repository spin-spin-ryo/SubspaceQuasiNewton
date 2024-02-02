from algorithms.descent_method import *
from algorithms.constrained_descent_method import *
from algorithms.proposed_method import *
from environments import *

def get_solver(solver_name,dtype):
    if solver_name == GRADIENT_DESCENT:
        solver = GradientDescent(dtype=dtype)
    elif solver_name == SUBSPACE_GRADIENT_DESCENT:
        solver = SubspaceGD(dtype=dtype)
    elif solver_name == ACCELERATED_GRADIENT_DESCENT:
        solver = AcceleratedGD(dtype=dtype)
    elif solver_name == NEWTON:
        solver = NewtonMethod(dtype=dtype)
    elif solver_name == LIMITED_MEMORY_NEWTON:
        solver = LimitedMemoryNewton(dtype=dtype)
    elif solver_name == PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingProximalGD(dtype=dtype)
    elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingAcceleratedProximalGD(dtype=dtype)
    elif solver_name == GRADIENT_PROJECTION:
        solver = GradientProjectionMethod(dtype=dtype)
    elif solver_name == DYNAMIC_BARRIER:
        solver = DynamicBarrierGD(dtype=dtype)
    elif solver_name == PRIMALDUAL:
        solver = PrimalDualInteriorPointMethod(dtype=dtype)
    elif solver_name == RSG_LC:
        solver = RSGLC(dtype=dtype)
    elif solver_name == RSG_NC:
        solver = RSGNC(dtype=dtype)
    else:
        raise ValueError(f"{solver_name} is not implemented.")
    return solver
    