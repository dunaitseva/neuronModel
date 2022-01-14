import numpy as np
import matplotlib.pyplot as plt

from ODE_solver.ODE_solve_methods import euler, implicit_euler, runge_kutta
from neuron_model import generate_reset_condition, generate_equation_system


def run_models(methods, _model_parameters, _end_time=300, _step=0.5, _I=5):
    """
    :param methods: List of functions, solving differential equation. This functions have to corresponds
                    the following signature method(x_0, start_time, end_time, func, step, additional_condition).
    :param _model_parameters:  Parameters, that set the neuron behaviour.
    :param _end_time: End of time integrating.
    :param _step: Step between points in time.
    :param _I: (Как там будет внешний ток по-английски???)
    :return: Return list of dictionaries,represents the running results in following format:
             [
                 {
                     't_nodes': t_solution,
                     'w_nodes': w_solution
                 },
                {
                     't_nodes': t_solution,
                     'w_nodes': w_solution
                 },
                 ...
             ]
    """

    # Initialize model running parameters and depends functions. a, b, c, d contains
    # parameters from _model_parameters. It using to generate other function such as
    # equation - parametrized differential equation of neuron model,
    # reset_condition - function that make a decision to reset the neuron value, when it
    # reaches a certain voltage.
    # Start value is required for solving the Cauchy problem.
    a, b, c, d, = _model_parameters
    equation = generate_equation_system(a, b, c, d, _I)
    reset_condition = generate_reset_condition(c, d)
    start_value = np.array([c, c * b])
    start_time = 0

    results = []

    for method in methods:
        t_sol, w_sol = method(start_value, start_time, _end_time, equation, _step, reset_condition)
        # methods support return solve in few dimensions, but we need only v(t) function solve values,
        # therefore we takes only w_sol[0]
        results.append({'t_nodes': t_sol, 'w_nodes': w_sol[0]})
        # results.append({'t_nodes': t_sol, 'w_nodes': w_sol})

    return results


def test_method(method):
    def eq(y, t):
        return y - t ** 2 + 1

    def solution(t):
        return (t + 1) ** 2 + (0.5 - 1) * np.exp(t)

    x_0 = 0.5
    t_0 = 0
    t_n = 2.5
    step = 0.5
    t_nodes = np.linspace(t_0, t_n, 100)
    y_nodes = solution(t_nodes)

    t_sol, y_sol = method(x_0, t_n, eq, step)

    plt.plot(t_nodes, y_nodes, color='red')
    plt.plot(t_sol, y_sol, color='blue')
    plt.legend(["y(t)", "w_i"])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    models_parameters = (
        (0.02, 0.2, -65, 6),  # Tonic Spiking
        (0.02, 0.25, -65, 6),  # Phasic Spiking
        (0.02, 0.2, -50, 2),  # Chattering
        (0.1, 0.2, -65, 2)  # Fast Spiking
    )
    model_using_step = 0.1

    figure_size = (14, 7)

    fig_tonic, axs_tonic = plt.subplots(sharex=True, figsize=figure_size)
    axs_tonic.set_title("Tonic spiking (TS)")

    fig_phasic, axs_phasic = plt.subplots(sharex=True, figsize=figure_size)
    axs_phasic.set_title("Phasic spiking (PS)")

    fig_chattering, axs_chattering = plt.subplots(sharex=True, figsize=figure_size)
    axs_chattering.set_title("Chattering (C)")

    fig_fast, axs_fast = plt.subplots(sharex=True, figsize=figure_size)
    axs_fast.set_title("Fast spiking (FS)")

    marker = 'o'
    linestyle = '--'
    markersize = 5.5
    linewidth = 1.5
    capacity = 0.5

    forward_euler_graphic_parameters = {
        'label': 'Euler',
        'marker': marker,
        'markersize': markersize,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'color': '#7A1E99',
        'alpha': capacity,
    }
    implicit_euler_graphic_parameters = {
        'label': 'Implicit Euler',
        'marker': marker,
        'markersize': markersize,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'color': '#009EA0',
        'alpha': capacity
    }
    runge_kutta_graphic_parameters = {
        'label': 'Runge-Kutta',
        'marker': marker,
        'markersize': markersize,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'color': '#F96B07',
        'alpha': capacity
    }

    tonic = run_models([euler, implicit_euler, runge_kutta], _model_parameters=models_parameters[0],
                       _step=model_using_step)
    tonic2 = run_models([euler, implicit_euler, runge_kutta], _model_parameters=models_parameters[0],
                       _step=0.1)
    axs_tonic.plot(tonic[0]['t_nodes'], tonic[0]['w_nodes'], **forward_euler_graphic_parameters)
    axs_tonic.plot(tonic[1]['t_nodes'], tonic[1]['w_nodes'], **implicit_euler_graphic_parameters)
    axs_tonic.plot(tonic[2]['t_nodes'], tonic[2]['w_nodes'], **runge_kutta_graphic_parameters)

    phasic = run_models([euler, implicit_euler, runge_kutta], _model_parameters=models_parameters[1],
                        _step=model_using_step)
    axs_phasic.plot(phasic[0]['t_nodes'], phasic[0]['w_nodes'], **forward_euler_graphic_parameters)
    axs_phasic.plot(phasic[1]['t_nodes'], phasic[1]['w_nodes'], **implicit_euler_graphic_parameters)
    axs_phasic.plot(phasic[2]['t_nodes'], phasic[2]['w_nodes'], **runge_kutta_graphic_parameters)

    chattering = run_models([euler, implicit_euler, runge_kutta], _model_parameters=models_parameters[2],
                            _step=model_using_step)
    axs_chattering.plot(chattering[0]['t_nodes'], chattering[0]['w_nodes'], **forward_euler_graphic_parameters)
    axs_chattering.plot(chattering[1]['t_nodes'], chattering[1]['w_nodes'], **implicit_euler_graphic_parameters)
    axs_chattering.plot(chattering[2]['t_nodes'], chattering[2]['w_nodes'], **runge_kutta_graphic_parameters)

    fast = run_models([euler, implicit_euler, runge_kutta], _model_parameters=models_parameters[3],
                      _step=model_using_step)
    axs_fast.plot(fast[0]['t_nodes'], fast[0]['w_nodes'], **forward_euler_graphic_parameters)
    axs_fast.plot(fast[1]['t_nodes'], fast[1]['w_nodes'], **implicit_euler_graphic_parameters)
    axs_fast.plot(fast[2]['t_nodes'], fast[2]['w_nodes'], **runge_kutta_graphic_parameters)

    axs_tonic.grid(which='both')
    axs_phasic.grid(which='both')
    axs_chattering.grid(which='both')
    axs_fast.grid(which='both')
    axs_tonic.legend(loc='best')
    axs_phasic.legend(loc='best')
    axs_chattering.legend(loc='best')
    axs_fast.legend(loc='best')

    fig_tonic.tight_layout()
    fig_tonic.savefig('../doc/static/TS.pdf', format='pdf')
    fig_tonic.savefig('../doc/static/TS.svg', format='svg')

    fig_phasic.tight_layout()
    fig_phasic.savefig('../doc/static/PS.pdf', format='pdf')
    fig_phasic.savefig('../doc/static/PS.svg', format='svg')

    fig_chattering.tight_layout()
    fig_chattering.savefig('../doc/static/C.pdf', format='pdf')
    fig_chattering.savefig('../doc/static/C.svg', format='svg')

    fig_fast.tight_layout()
    fig_fast.savefig('../doc/static/FS.pdf', format='pdf')
    fig_fast.savefig('../doc/static/FS.svg', format='svg')
