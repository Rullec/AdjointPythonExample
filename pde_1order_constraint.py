import numpy as np
tar_pos = -4
'''
Adjoint method example for learning

We use adjoint method to solve a PDE-constrained optimization:

    min_p f(x, p)
    s.t. g(x, p) = 0

    constraint: g(x, p) = x' + p.x + 1 = 0, where :
        x = x(t)
        x' = dx/dt
        Init Contition: x(0) = 0
        
        Though the analytic solution of this ODE is: x(t) = -(1 + e^{-pt}) / p

        We choose to solve this PDE by numerical method.

    energy: f(x, p) = |x(0.5) - tar_pos|^2

    discretization:
    
    df/dp = 2 * (x(0.5) - tar_pos) * dx(0.5) /dp
    x_next = (1 - dt * p) * xt - dt
    dx_{t+1}/dp = - [ (dt * p - 1.0) * dxt/dp + dt * xt]
'''

def do_simulation(x0, dt, tar_frame, p):
    x_lst = [x0]
    dxdp_lst = [0]
    dxdp_cur = 0
    x_cur = x0
    for i in range(tar_frame):
        x_next = (1 - dt * p) * x_cur  - dt
        x_lst.append(x_next)
        dxdp_cur = - ( (dt * p - 1.0) * dxdp_cur + dt * x_cur)
        dxdp_lst.append(dxdp_cur)
        x_cur = x_next
    return x_lst, dxdp_lst

def energy(x_final):
    return (x_final - tar_pos) ** 2
if __name__ == "__main__":
    p = 0
    max_iters = 100
    dt = 0.01
    target_timept = 0.5
    tf = int( target_timept / dt)
    x0 = 0
    alpha= 0.1

    for iter in range(max_iters):
        # 1. do forward simulation, calculate x[t] and dx[t]/dp, calculate energy
        x_lst, dxdp_lst = do_simulation(x0, dt, tf, p)
        x_final = x_lst[-1]
        dxdp_final = dxdp_lst[-1]

        cur_E = energy(x_final)
        
        # 2. calculate df/dp
        '''
        df/dp = 2(x[tf] - 2) * dx[tf]/dp

        dx[t+1]/dp = - [ (p - 1.0 / dt) * dx[t]/dp + x[t]]
        dx[0]/dp = 0
        '''
        dfdp = 2 * (x_final - tar_pos) * dxdp_final

        # 3. update
        p -= alpha * dfdp
        
        if iter % 10 == 0:
            print(f"iter {iter} cur p {p:.5f} x_final {x_final:.5f} energy {cur_E:.5f}, dfdp {dfdp:.5f}")