import numpy as np
import matplotlib.pyplot as plt

from ODE_solver.ODE_solve_methods import implicit_euler, euler, runge_kutta


def foo(w, t):
    return w - t * t + 1


def exact_solution(t):
    return (t + 1) * (t + 1) + (0.5 - 1) * np.exp(t)


if __name__ == '__main__':
    exact_visualisation = {
        'linestyle': '-',
        'linewidth': 2,
        'color': 'blue',
        'alpha': 0.5
    }

    solution_visualization = {
        'marker': 'o',
        'markersize': 3,
        'linestyle': '--',
        'linewidth': 3,
        'alpha': 0.5
    }

    start_time = 0
    end_time = 26
    step = 0.6

    t_nodes = np.arange(start_time, end_time + step, step)
    y_nodes = exact_solution(t_nodes)

    solution_euler = euler(x_0=0.5, start_time=start_time, end_time=end_time, func=foo, step=step)
    solution_implicit = implicit_euler(x_0=0.5, start_time=start_time, end_time=end_time, func=foo, step=step)
    solution_runge = runge_kutta(x_0=0.5, start_time=start_time, end_time=end_time, func=foo, step=step)

    plt.yscale("log")
    # plt.plot(t_nodes, y_nodes, **exact_visualisation)
    plt.plot(solution_euler[0], np.abs(solution_euler[1] - y_nodes), color='red', label='euler', **solution_visualization)
    plt.plot(solution_implicit[0], np.abs(solution_implicit[1] - y_nodes), color='green', label='impl', **solution_visualization)
    plt.plot(solution_runge[0], np.abs(solution_runge[1] - y_nodes), color='cyan', label='runge', **solution_visualization)
    plt.grid(True)
    plt.legend()
    plt.show()
