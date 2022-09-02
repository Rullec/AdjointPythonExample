import numpy as np
import os
import matplotlib.pyplot as plt
import os.path as osp
tar_pos = -2.0
'''
Adjoint method example for learning

We use adjoint method to solve a PDE_constrained optimization:

    min_p   f(x, p)
    s.t.    g(x, p) = 0
    
    constraint: g(x, p) = x'' + p[0]x' + p[1]x + 1 = 0
        x = x(t)
        x'' = d^2 x / dt^2
        x' = dx/dt
        Init condition x(0) = 0, x'(0) = 0

    energy:     f(x, p) = |x(0.5) - tar_pos|^2

    discretization:
        x''_t = (x_{t+1} - 2 x_t + x_{t-1}) / dt^2
        x'_t = (x_{t+1} - x_t) / dt

        
        g_t(x_t, p) = x''_t + p0 * x'_t + p_1 * x_t + 1 = 0
        \iff
        g_t(x_{t+1}, x_t, x_{t-1}, p) = x_{t+1} (1 + dt * p0) + xt * (-2 -dt * p0 + p1 * dt^2) + x_{t-1} + dt*2 = 0
        
        x_{t+1}  = 
             -  [ xt * (-2 -dt * p0 + p1 * dt^2) + x_{t-1} + dt*2 ]
                / 
                (1 + dt * p0)


    Calc Derivatives:
    
        df/dp = \partial f / \partial p + \partial f/\partial x(0.5) * d x(0.5)/dp
        \partial f / \partial p = 0
        \partial f / \partial x(0.5) = 2(x(0.5) - tar_pos)

    Adjoint:
        dg_t / dp   =   \partial g_t / \partial p
                    +   \partial g_t / \partial x_{t+1} * d(x_{t+1}) / dp
                    +   \partial g_t / \partial x_{t} * d(x_{t}) / dp
                    +   \partial g_t / \partial x_{t-1} * d(x_{t-1}) / dp
                    = 0
        and
        \partial g_t / \partial p = [dt * x_{t+1} - dt * xt, dt^2 * xt]
        \partial g_t / \partial x_{t+1} = 1 + dt * p0
        \partial g_t / \partial x_t = p_1 * dt^2 - dt * p0 - 2
        \partial g_t / \partial x_{t-1} = 1

        so
        d(x_{t+1}) / dp = 
            -(  \partial g_t / \partial p 
                + \partial g_t / \partial x_{t} * d(x_{t}) / dp 
                + \partial g_t / \partial x_{t-1} * d(x_{t-1}) / dp
            ) / 
            (\partial g_t / \partial x_{t+1})


'''

def do_simulation(x0, xdot0, dt, tf, p):
    p0 = p[0]
    p1 = p[1]
    
    
    # (xcur - xpre)/dt = xdot0
    # xpre = xcur - dt * xdot0
    xcur = x0
    xpre = xcur - dt * xdot0
    x_lst = [x0]

    dxcur_dp = 0
    dxpre_dp = 0
    
    for i in range(tf):
        # 1. update x
        '''
        x_{t+1}  = 
            -  [ xt * (-2 -dt * p0 + p1 * dt^2) + x_{t-1} + dt*2 ]
            / 
            (1 + dt * p0)
        '''
        xnext = - (xcur * (-2 - dt * p0 + p1 * dt * dt) + xpre + dt * dt) / (1 + dt * p0)

        # 2. update gradient
        '''
        d(x_{t+1}) / dp = 
            -(  \partial g_t / \partial p 
                + \partial g_t / \partial x_{t} * d(x_{t}) / dp 
                + \partial g_t / \partial x_{t-1} * d(x_{t-1}) / dp
            ) / 
            (\partial g_t / \partial x_{t+1})
        '''
        '''
        dgtdp = \partial g_t / \partial p = [dt * x_{t+1} - dt * xt, dt^2 * xt]
        dgtdxpre = \partial g_t / \partial x_{t-1} = 1.0
        dgtdxcur = \partial g_t / \partial x_{t} = p_1 * dt^2 - dt * p0 - 2
        dgtdxnext = \partial g_t / \partial x_{t+1} = 1 + dt * p0
        '''
        dgtdp = [dt * xnext - dt * xcur, dt * dt * xcur]
        dgtdxpre = 1.0
        dgtdxcur = p1 * dt * dt - dt * p0 - 2
        dgtdxnext = 1 + dt * p0

        dxnext_dp = - (dgtdp + dgtdxcur * dxcur_dp + dgtdxpre * dxpre_dp) / dgtdxnext

        # 3. push into the list
        x_lst.append(xnext)

        xpre = xcur
        xcur = xnext

        dxpre_dp = dxcur_dp
        dxcur_dp = dxnext_dp
    
    return x_lst, dxnext_dp


def energy(x_final):
    return (x_final - tar_pos) ** 2

if __name__ == "__main__":
    p = np.array([0.3, 0.1])
    max_iters = 100000
    output_iter = 100
    dt = 1e-2
    target_timept = 0.5
    tf = int(target_timept / dt)
    x0 = 0
    xdot0 = 0
    alpha = 1

    png_output_dir = "output"
    if osp.exists(png_output_dir) == False:
        os.makedirs(png_output_dir)
    
    for iter in range(max_iters):
        # 1. do forward simualtion calcute x[t] and dx[t]/dp
        x_lst, dxdp_final = do_simulation(x0, xdot0, dt, tf, p)

        # save current solution img
        plt.cla()
        t_lst = np.linspace(0, target_timept, len(x_lst))
        plt.plot(t_lst, x_lst)
        plt.xlim([0, target_timept + 0.1])
        plt.ylim([tar_pos * 1.1, 0])
        plt.scatter([target_timept], [tar_pos], s = 50, color = 'red')
        plt.title(f"iter {iter}")
        plt.savefig(f"{png_output_dir}/{iter}.png")

        # 2. calculate f and df/dp
        '''
        df/dp = 2(x(0.5) - tar_pos) * dxdp_final
        '''
        x_final = x_lst[-1]
        cur_E = energy(x_final)
        dfdp = 2 * (x_final - tar_pos) * dxdp_final
        # print(f"dxdp_final {dxdp_final}")
        # print(f"dfdp {dfdp}")
        # 3. update
        p -= alpha * dfdp
    
        should_stop = cur_E < 1e-6
        should_output = should_stop== True or iter % output_iter == 0
        if should_output:
            print(f"iter {iter} cur p {p[0]:.3f} {p[1]:.3f} x_final {x_final:.3f} energy {cur_E:.5f}, dfdp {dfdp[0]:.3f} {dfdp[1]:.3f}")
        if should_stop:
            print(f"optimization done! x_final {x_final:.3f} ~= target pos {tar_pos}, current gradient is {dfdp}, current energy is {cur_E:.3e}")
            break