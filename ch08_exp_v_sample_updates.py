import numpy as np
import matplotlib.pyplot as plt


# --------------------
# Figure 8.8: Comparison of efficiency of expected and sample updates.
# --------------------

def fig_8_8():
    #
    # Note key assumptions:
    #   -- b successor states are equally likely
    #   -- error in initial estimate is 1
    #   -- values at next state assumed correct
    # Then:
    #   -- expected update reduces the error to 0 in expectation upon its completion
    #       (ie at number of updates equal to the branches)
    #   -- sample updates reduce the error by a factor of sqrt((b-1) / (b*t)) after each update; specifically:
    #           1. (b-1)/b is the factor of reduction (since value at next state assumed correct)
    #           2. 1/t where t is number of updates assuming sample average ie alpha = 1/t
    #               the MSE(sample mean) = var / num_samples; if we assume the branches sample a Normal(0,1) and var = 1, 
    #               then the MSE(sample mean) = 1 / num_samples = 1 / t
    #               thus MSE(sample mean) reduction factor is (b-1)/b * 1/t


    fig = plt.figure()
    plt.xlim(-0.02,2.02)
    plt.xticks(range(3))
    plt.gca().set_xticklabels(['0', '1b', '2b'])
    plt.xlabel(r"Number of $\max_{a'}$ Q(s',a') computations")
    plt.ylim(-0.02,1.02)
    plt.yticks([0,1])
    plt.ylabel('RMS error in value estimate')

    # plot expected updates
    x = np.linspace(0,2,1000)
    expected_rms = np.array(x <= 1, dtype=np.float)
    plt.plot(x, expected_rms, c='gray', label='Expected updates', lw=1)

    # plot the sample updates at various branching factors
    b_range = [2, 10, 100, 1000, 10000]
    for b in b_range:
        # the plotting coordinates
        x = np.linspace(1e-8, 2, min(1000, 2*b+1))

        # the number of sample updates for the error reduction function
        t = np.linspace(1e-8, 2*b, min(1000, 2*b+1))

        # Sec 8.5 -- sample update reduce the error by sqrt((b-1) / (b*t))
        # the starting rms error is 1 so only need to plot the reduction factor,
        # evaluated at the various branching factors b and number of updates in [0, 2*b]
        sample = np.sqrt((b-1)/(b*t))

        # only take the values in the plotting y-range [0,1] so it matches the Fig 8.8
        xx = x[np.where(sample < 1)]
        sample = sample[np.where(sample < 1)]

        plt.plot(xx, sample, label='Sample updates b={}'.format(b), lw=1)

    plt.legend()

    plt.savefig('figures/ch08_fig_8_8.png')
    plt.close()


if __name__ == '__main__':
    fig_8_8()

