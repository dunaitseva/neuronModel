import numpy as np
import matplotlib.pyplot as plt


def simulate_network(exc=800, inh=200, step=0.5, threshold=30., end_time=1000.):
    rnd = np.random.uniform
    a = np.hstack((0.02 * np.ones(exc), 0.02 + 0.08 * rnd(size=inh)))
    b = np.hstack((0.2 * np.ones(exc), 0.25 - 0.05 * rnd(size=inh)))
    c = np.hstack((-65. + 15. * rnd(size=exc) ** 2., -65. * np.ones(inh)))
    d = np.hstack((8. - 6. * rnd(size=exc) ** 2., 2. * np.ones(inh)))

    v = -65. * np.ones_like(a)
    u = b * v

    W = np.hstack((0.5 * rnd(size=(exc + inh, exc)), -rnd(size=(exc + inh, inh))))

    def update_neurons(_I):
        _v = v + step * (0.04 * v ** 2. + 5. * v + 140. - u + _I)
        _u = u + step * (a * (b * v - u))
        # return _v + step * (0.04 * _v ** 2. + 5. * _v + 140. - _u + I), _u + step * (a * (b * _v - _u))
        return _v, _u

    registered_spikes = []
    exc_cur_mult = rnd(size=exc)
    inh_cur_mult = rnd(size=inh)
    for t in np.arange(end_time, step=step):
        firing = v >= threshold
        v[firing] = c[firing]
        u[firing] = u[firing] + d[firing]
        _fired = np.where(firing)
        if len(_fired[0]) > 0:
            registered_spikes.append({'time': t, 'indices': _fired[0]})

        I = np.hstack((5 * exc_cur_mult, 2 * inh_cur_mult))
        I += np.sum(W[:, firing], axis=1)

        v, u = update_neurons(I)

    return registered_spikes


def visualize_neuron_network(registered_spikes, axes, *args, **kwargs):
    for time_node in registered_spikes:
        for neuron_index in time_node['indices']:
            axes.plot([time_node['time']], [neuron_index], **kwargs)


if __name__ == '__main__':
    network_working_computations = simulate_network(800, 200, step=0.5, end_time=1000.)

    figure_size = (14, 7)
    fig, axs = plt.subplots(sharex=True, figsize=figure_size)
    axs.set_title("Simulation of a network of 1000 randomly coupled spiking neurons.")

    marker = 'o'
    linestyle = '--'
    markersize = 5.5
    linewidth = 1.5
    capacity = 0.2

    graphic_parameters = {
        'marker': marker,
        'markersize': markersize,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'color': '#009EA0',
        'alpha': capacity,
    }

    visualize_neuron_network(network_working_computations, axs, **graphic_parameters)
    axs.grid(which='both')
    axs.set_xlabel('time')
    axs.set_ylabel('neuron number')
    fig.savefig('../doc/static/network.pdf', format='pdf')
