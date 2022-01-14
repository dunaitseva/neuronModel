import numpy as np
from scipy.optimize import root, fsolve


# def euler(x_0, end_time, func, step):
#     """
#     :param x_0: Initial value for the Cauchy problem
#     :param end_time: End of tome integrating.
#     :param func: Function, has signature func(w_i: float, t_i: float). Equation of normalized ODE.
#                  Function calculate value of equation for dy/dt at moment t_i according to Euler's formula.
#                  w_i is approximate value y(t_i).
#     :param step: step between points in time.
#     :return: Two lists. First - time nodes, second - approximate values in current moment time.
#     """
#
#     t_nodes = np.linspace(0, end_time, int(end_time // step))[1:]
#
#     # According Cauchy problem, set start value for
#     # first node w_0
#     # The following code corresponds to the actions in the euler method
#     w_nodes = [x_0]
#
#     def w_next(w_prev, t_node):
#         print(func(w_prev, t_node))
#         return w_prev + step * func(w_prev, t_node)
#
#     for i, t in enumerate(t_nodes[1:]):
#         w_nodes.append(w_next(w_nodes[i], t))
#
#     return t_nodes, w_nodes

def default_additional_condition(w):
    return w


def euler(x_0, start_time, end_time, func, step, additional_condition=default_additional_condition):
    """
    :param x_0: Initial value for the Cauchy problem
    :param end_time: End of tome integrating.
    :param func: Function, has signature func(w_i: float, t_i: float). Equation of normalized ODE.
                 Function calculate value of equation for dy/dt at moment t_i according to Euler's formula.
                 w_i is approximate value y(t_i).
    :param step: step between points in time.
    :param additional_condition: is the function, that imposes additional restrictions on w_i value
    :return: Two lists. First - time nodes, second - approximate values in current moment time.
             Values correspond dimensions of x_0 vector
    """

    t_nodes = np.arange(start_time, end_time + step, step=step)

    # The following code corresponds to the actions in the euler method
    # According Cauchy problem, set start value for first node w_0
    w_nodes = [additional_condition(np.array(x_0))]

    def w_next(w_prev, t_node):
        return w_prev + step * func(w_prev, t_node)

    for i, t in enumerate(t_nodes[:-1]):
        w_nodes.append(additional_condition(w_next(w_nodes[i], t)))

    return t_nodes, np.array(w_nodes).transpose()


def implicit_euler(x_0, start_time, end_time, func, step, additional_condition=default_additional_condition):
    """
    :param x_0: Initial value for the Cauchy problem.
    :param start_time: Start of tome integrating.
    :param end_time: End of tome integrating.
    :param func: Function, has signature func(w_i: float, t_i: float). Equation of normalized ODE.
                 Function calculate value of equation for dy/dt at moment t_i according to Euler's formula.
                 w_i is approximate value y(t_i).
    :param step: step between points in time.
    :param additional_condition: is the function, that imposes additional restrictions on w_i value
    :return: Two lists. First - time nodes, second - approximate values in current moment time.
             Values correspond dimensions of x_0 vector.
    """

    t_nodes = np.arange(start_time, end_time + step, step=step)

    # The following code corresponds to the actions in the implicit euler method
    # According Cauchy problem, set start value for first node w_0
    w_nodes = [additional_condition(np.array(x_0))]

    def w_next(w_prev, t_node):
        return w_prev + step * func(w_prev, t_node)

    for i, t in enumerate(t_nodes[1:]):
        # trick as predictor-corrector method
        # using only for experiments or for implementation
        # in other ODE solver functions.
        # next_val = w_next(w_nodes[i], t)
        # next_val = w_nodes[i] + step * func(next_val, t)
        # w_nodes.append(additional_condition(next_val))
        solution = root(
            lambda w_next: w_nodes[i] + step * func(w_next, t) - w_next,
            w_nodes[i],
            options={'maxfev': 10000}
        )
        w_nodes.append(additional_condition(solution.x))

    return t_nodes, np.asarray(w_nodes, dtype=object).transpose()


def runge_kutta(x_0, start_time, end_time, func, step, additional_condition=default_additional_condition):
    """
    :param x_0: Initial value for the Cauchy problem.
    :param start_time: Start of tome integrating.
    :param end_time: End of tome integrating.
    :param func: Function, has signature func(w_i: float, t_i: float). Equation of normalized ODE.
                 Function calculate value of equation for dy/dt at moment t_i according to Euler's formula.
                 w_i is approximate value y(t_i).
    :param step: step between points in time.
    :param additional_condition: is the function, that imposes additional restrictions on w_i value
    :return: Two lists. First - time nodes, second - approximate values in current moment time.
             Values correspond dimensions of x_0 vector.
    """

    t_nodes = np.arange(start_time, end_time + step, step=step)

    # The following code corresponds to the actions in the Runge-Kutta method
    # According Cauchy problem, set start value for first node w_0
    w_nodes = [additional_condition(np.array(x_0))]

    def k_1(w_prev, t_node):
        return step * func(w_prev, t_node)

    def k_2(w_prev, t_node):
        return step * func(w_prev + 0.5 * k_1(w_prev, t_node), t_node + step * 0.5)

    def k_3(w_prev, t_node):
        return step * func(w_prev + 0.5 * k_2(w_prev, t_node), t_node + step * 0.5)

    def k_4(w_prev, t_node):
        return step * func(w_prev + k_3(w_prev, t_node), t_node + step)

    for i, t in enumerate(t_nodes[:-1]):
        w = w_nodes[i]
        w_next = w + (1 / 6) * (k_1(w, t) + 2 * k_2(w, t) + 2 * k_3(w, t) + k_4(w, t))
        w_nodes.append(additional_condition(w_next))

    return t_nodes, np.asarray(w_nodes, dtype=object).transpose()
