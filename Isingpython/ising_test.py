import os
import numpy as np
import csv
from ising_feed_forward import test_both, execute_feed_forward
from ising_convolutional import run_no_neptune
from ising_convolutional import PARAMS as PARAMS_CONVO
from ising_feed_forward import PARAMS as PARAMS_FFD
from ising import IsingData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BATCH_50 = [
    "out/50_rel_point05_1mil/meta_50_rel_point05_1mil.json",
    "out/50_rel_point07_1mil/meta_50_rel_point07_1mil.json"
]


class IsingTest:

    @staticmethod
    def feed_forward_test_width_depth(head, tail):
        return test_both(head, tail)

    @staticmethod
    def feed_forward_test(head, tail, use_max=False):
        return execute_feed_forward(head, tail, use_max=use_max)

    @staticmethod
    def get_overlap(head, tail):
        data = IsingData()
        data.load_json(tail)
        return data.plot_energy_spectrum(), data.plot_magnetization_spectrum()

    @staticmethod
    def test_convo(head, tail):
        min_depth = 1
        max_depth = 4
        min_start_filter = 2
        max_start_filter = 10

        n_step_filter = 5
        n_step_depth = 4

        depths = np.linspace(min_depth, max_depth, n_step_depth, dtype=np.int32)
        filters = np.linspace(min_start_filter, max_start_filter, n_step_filter, dtype=np.int32)

        dv, fv = np.meshgrid(depths, filters, indexing="ij")

        acc = np.zeros((n_step_depth, n_step_filter))

        for i in range(n_step_depth):
            for j in range(n_step_filter):
                PARAMS_CONVO["conv_depth"] = dv[i, j]
                PARAMS_CONVO["conv_start_filters"] = fv[i, j]
                acc[i, j] = run_no_neptune(head, tail)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("Convolutional Depth")
        ax.set_ylabel(f"Convolutional Start filters - increment {PARAMS_CONVO['conv_increment']}")
        ax.set_zlabel("Accuracy")

        ax.plot_surface(dv, fv, acc, cmap="jet")
        plt.savefig("3D convolutional test.png")
        plt.show()
        return acc

    def __init__(self, batch, tasks, kwargs=None, execute=True):
        self.batch = batch
        self.tasks = tasks
        if kwargs:
            self.kwargs = kwargs
            assert len(self.kwargs) == len(self.tasks)
        self.res = np.array([0] * len(self.batch) * len(self.tasks))
        self.res = self.res.reshape((len(self.batch), len(self.tasks)))
        if execute:
            self.execute()

    def execute(self):
        BASE = os.getcwd()
        for i, batch in enumerate(self.batch):
            head, tail = os.path.split(batch)
            os.chdir(os.path.join(BASE, head))
            for j, task in enumerate(self.tasks):
                self.res[i, j] = task(head, tail, **self.kwargs[j])

    def write_csv(self, filename):
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["file"].extend([t.__name__ for t in self.tasks]))
            for i, batch in enumerate(self.batch):
                csv_writer.writerow([self.batch[i]].extend(self.res[i]))


if __name__ == "__main__":
    pass
