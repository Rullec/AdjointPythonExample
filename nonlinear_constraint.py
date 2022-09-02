import numpy as np
'''
Adjoint method example for learning

We use adjoint method to solve a constraint optimization:

    min_p f(x, p)
    s.t. g(x, p) = 0

f(x, p) = x.dot(x) + p.dot(p)
g(x, p) = 3 * x + p - 5
'''


def constraint(x, p):
    return 3 * x + p - 5


def recalc_x_from_p(p):
    return (5 - p) / 3


def constraint_dx(x, p):
    return 3 * np.identity(len(x))

def constraint_dp(x, p):
    return np.identity(len(p))


def energy(x, p):
    return np.dot(x.T, x) + np.dot(p.T, p)


def energy_dx(x, p):
    return 2 * x.T


def energy_dp(x, p):
    return 2 * p.T


def calc_numeric_grad_x(eval_func, x, p, eps=1e-5):
    x_dim = len(x)
    old_E = eval_func(x, p)
    if type(old_E) is float:
        old_E = np.array(old_E)

    num_grad = np.zeros([len(old_E), x_dim])
    for i in range(x_dim):
        x[i] += eps
        new_E = eval_func(x, p)
        x[i] -= eps

        num_grad[:, i] = (new_E - old_E) / eps
    return num_grad


def calc_numeric_grad_p(eval_func, x, p, eps=1e-5):

    old_E = eval_func(x, p)
    if type(old_E) == float:
        old_E = np.array([old_E])

    # print(f"old E {old_E} len(p) {len(p)}")
    num_grad = np.zeros([len(old_E), len(p)])
    for i in range(len(p)):
        p[i] += eps
        new_E = eval_func(x, p)
        p[i] -= eps
        num_grad[:, i] = (new_E - old_E) / eps

    return num_grad


def check_gradient(func, func_ana_grad, x, p, func_num_grad):
    eps = 1e-5
    ana = func_ana_grad(x, p)
    num = func_num_grad(func, x, p, eps)
    assert ana.shape == num.shape, f"ana grad shape {ana.shape} != num grad shape {num.shape}"
    diff = ana - num

    max_diff = np.max(np.abs(diff))
    assert max_diff < 10 * eps, f"max diff {max_diff} >= {10 * eps}, check failed. ana {ana} num {num} "
    return True


def calc_adjoint_grad(x, p):
    '''
    df/dp = f_p - f_x * (g_x)^{-1} * g_p
    '''
    fp = energy_dp(x, p)
    fx = energy_dx(x, p)
    gxinv = np.linalg.inv(constraint_dx(x, p))
    gp = constraint_dp(x, p)
    grad = fp - np.matmul(np.matmul(fx, gxinv), gp)
    
    return grad


if __name__ == "__main__":
    # 1. init
    x_dim = 2
    p_dim = 2

    # 2. check the partial gradient is correctly implemented

    p0 = np.array(np.random.rand(p_dim, 1))

    x0 = recalc_x_from_p(p0)
    assert check_gradient(energy, energy_dp, x0, p0, calc_numeric_grad_p)
    assert check_gradient(energy, energy_dx, x0, p0, calc_numeric_grad_x)
    assert np.max(np.abs(constraint(x0, p0))) < 1e-5
    # 3. do iterations
    max_iters = 200
    alpha = 1e-2
    output_iter = 10
    for iter in range(max_iters):
        cur_E = np.squeeze( energy(x0, p0))
        
        dir = calc_adjoint_grad(x0, p0).T
        
        if iter % output_iter == 0:
            print(f"iter {iter} p0 {p0.T} dir {dir.T} Energy = {cur_E:.3f}")

        p0 -= alpha * dir
        x0 = recalc_x_from_p(p0)
