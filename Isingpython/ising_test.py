import os
import numpy as np
from ising_feed_forward import test_both, execute_feed_forward
from ising_convolutional import run_neptune
from ising_convolutional import PARAMS as PARAMS_CONVO
from ising_feed_forward import PARAMS as PARAMS_FFD
from ising import IsingData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_convo(head, tail):
    min_depth = 2
    max_depth = 10
    min_start_filter = 2
    max_start_filter = 10

    n_step_filter = 5
    n_step_depth = 5

    depths = np.linspace(min_depth, max_depth, n_step_depth)
    filters = np.linspace(min_start_filter, max_start_filter, n_step_filter)

    dv, fv = np.meshgrid(depths, filters, indexing="ij")

    acc = np.zeros((n_step_depth, n_step_filter))

    for i in range(n_step_filter):
        for j in range(n_step_depth):
            PARAMS_CONVO["conv_depth"] = depths[i, j]
            PARAMS_CONVO["conv_start_filters"] = filters[i, j]
            acc[i, j] = run_neptune(head, tail)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Convolutional Depth")
    ax.set_ylabel(f"Convolutional Start filters - increment {PARAMS_CONVO['conv_increment']}")
    ax.set_zlabel("Accuracy")

    ax.plot_surface(dv, fv, acc, cmap="jet")
    plt.savefig("3D convolutional test.png")
    plt.show()
    return acc


if __name__ == "__main__":
    file = input("Enter JSON file to test")
    head, tail = os.path.split(file)
    os.chdir(os.path.join(os.getcwd(), head))
    print(f"Test feed forward")
    ffacc = execute_feed_forward(head, tail, use_max=False, plotspectrum=False, runneptune=True)
    print("Test convolutional (not periodic)")
    PARAMS_CONVO["periodic_padding"] = False
    conv_acc = run_neptune(head, tail)
    print("Test convolutional (periodic)")
    PARAMS_CONVO["periodic_padding"] = True
    conv_acc_p = run_neptune(head, tail)
    print(
        f"Feed forward accuracy:{ffacc[1]}\nConvolutional accuracy (no padding):{conv_acc}\nConvolutional accuracy (padding):{conv_acc_p}")
