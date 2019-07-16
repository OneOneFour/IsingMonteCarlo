import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from time import gmtime, strftime
import os

T_CRIT = 2 / np.log(1 + np.sqrt(2))


class IsingLattice:
    def __init__(self, size, kT, cold=True):
        self.size = size
        self.kT = kT
        self.critical = bool(self.kT > 2 / np.log(1 + np.sqrt(2)))
        if cold:
            self.lattice = [[1 for i in range(self.size)] for j in range(self.size)]
        else:
            self.lattice = [[random.choice((-1, 1)) for i in range(self.size)] for j in range(self.size)]

        self.magnetization = [self.cur_magnetization()]
        self.energy = [self.energy_periodic()]
        self.record_states = []

    def energy_periodic(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):
                E += -1 * self.lattice[y][x] * (
                        self.lattice[(y + 1) % self.size][x] +
                        self.lattice[(y - 1) % self.size][x] +
                        self.lattice[y][(x + 1) % self.size] +
                        self.lattice[y][(x - 1) % self.size]
                )
        return E

    def energy_free_ends(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):  # Use nearest neighbouring 4 cells
                E += self.lattice[y][x] * (
                        (self.lattice[y + 1][x] if self.lattice[y + 1][x] < self.size else 0) +
                        (self.lattice[y - 1][x] if self.lattice[y - 1][x] >= 0 else 0) +
                        (self.lattice[y][x + 1] if self.lattice[y + 1][x] < self.size else 0) +
                        (self.lattice[y][x - 1] if self.lattice[y + 1][x] >= 0 else 0)
                )
        return E

    def showlattice(self):
        fig, ax = plt.subplots()
        ax.imshow(self.lattice, cmap='jet')
        plt.show()

    def cur_magnetization(self):
        magnetization = 0
        for row in self.lattice:
            for spin in row:
                magnetization += spin
        return magnetization / (self.size ** 2)

    def abs_cur_magnetization(self):
        magnetization = 0
        for row in self.lattice:
            for spin in row:
                magnetization += spin
        return np.abs(magnetization / (self.size ** 2))
        # Flip the configuration spin

    def __wolff_step(self, f=0, im=None):
        '''
        Work using bonds as opposed to individual spins on the lattice.
        Converts problem to a percolation problem which unlike Metropolis does not fall in to inf relaxation time at the critical
        temperature.
        :return: Artist object to be blitted to matplotlib canvas
        '''
        rand_y, rand_x = np.random.randint(0, self.size, size=2)
        cluster = [(rand_y, rand_x)]
        deltaE = 0

        def check_and_add(y, x, y_1, x_1, deltaE):
            y_1, x_1 = (y_1 % self.size, x_1 % self.size)
            if (y_1, x_1) in cluster:
                return
            if self.lattice[y][x] == self.lattice[y_1][x_1]:
                if random.random() <= (1 - np.exp(-1 / self.kT)):
                    cluster.append((y_1, x_1))
                    deltaE += 2 * self.lattice[y_1][x_1] * (  # This calculation is CORRECT, don't let anyone tell you otherwise!
                            self.lattice[(y_1 + 1) % self.size][x_1] +
                            self.lattice[(y_1 - 1) % self.size][x_1] +
                            self.lattice[y_1][(x_1 + 1) % self.size] +
                            self.lattice[y_1][(x_1 - 1) % self.size])

        for p in cluster:
            check_and_add(p[0], p[1], p[0] + 1, p[1], deltaE)
            check_and_add(p[0], p[1], p[0] - 1, p[1], deltaE)
            check_and_add(p[0], p[1], p[0], p[1] + 1, deltaE)
            check_and_add(p[0], p[1], p[0], p[1] - 1, deltaE)

        for p in cluster:
            self.lattice[p[0]][p[1]] *= -1

        self.energy.append(self.energy[-1] + deltaE)  # weight these samples by the MC time occupied by each one
        self.magnetization.append(self.abs_cur_magnetization())
        if im:
            im.set_data(self.lattice)
            return im,
        else:
            return len(cluster)

    def __metropolis_step(self, f=0, im=None, max_iter=5000, batch=1):
        for i in range(batch):
            if im:
                if f >= max_iter - 1:
                    plt.close()
                # print(f"\rFrame:{f} of 5000", end='')
            rand_y, rand_x = np.random.randint(0, self.size, size=2)
            deltaE = 2 * self.lattice[rand_y][rand_x] * (
                    self.lattice[(rand_y + 1) % self.size][rand_x] +
                    self.lattice[(rand_y - 1) % self.size][rand_x] +
                    self.lattice[rand_y][(rand_x + 1) % self.size] +
                    self.lattice[rand_y][(rand_x - 1) % self.size]
            )
            r = random.random()
            w = np.exp((-1 / self.kT) * deltaE)
            if deltaE <= 0 or r <= w:
                self.lattice[rand_y][rand_x] *= -1
                self.magnetization.append(
                    self.magnetization[-1] + 2 * self.lattice[rand_y][rand_x] / (self.size ** 2))
                self.energy.append(self.energy[-1] + deltaE)
            else:
                self.magnetization.append(self.magnetization[-1])
                self.energy.append(self.energy[-1])
        if im:
            im.set_data(self.lattice)
            return im,
        else:
            return batch

    def start(self, method='metropolis', max_iter=5000, export_every=0, delay=0, log_correlation=False):
        if log_correlation:
            corrcoeff = []
        if export_every != 0:
            self.record_states = [0] * (int(max_iter / export_every) - 1)
        i = 0
        pbar = tqdm(total=max_iter)
        while i < max_iter:
            if method == 'metropolis':
                steps = self.__metropolis_step()
            elif method == 'wolff':
                steps = self.__wolff_step()
            else:
                raise ValueError(f"{method} is not a supported iteration method. Please choose from 'wolff' or 'metropolis' ")

            if export_every != 0 and i >= delay:
                if int(i / export_every) >= len(self.record_states):
                    # capture snapshot of the image
                    if log_correlation
                        if len(corrcoeff) > 0:
                            plt.plot(corrcoeff, label=f"{i}")
                            corrcoeff = []
                    self.record_states[int(i / export_every)] = pickle.loads(pickle.dumps(self.lattice))
            if log_correlation:
                 if len(self.record_states) > 0:
                    corrcoeff.append(np.corrcoef(np.array(self.lattice).flatten(), np.array(self.record_states[-1]).flatten())[0][1])

            # Update progress bar and loop progress
            i += steps
            pbar.update(steps)

        self.energy = self.energy[delay:]
        self.magnetization = self.magnetization[delay:]
        if log_correlation:
            plt.title(f"R^2 correlation at T:{self.kT}")
            plt.legend()
            plt.show()

        return self.__dict__

    def start_animation(self, max_iter=500000, batch=1):
        import matplotlib.animation as animation
        fig, ax = plt.subplots()

        im = plt.imshow(self.lattice, cmap='jet', animated=True, vmin=-1, vmax=1)

        animation.FuncAnimation(fig, self.__wolff_step, fargs=(im,), blit=True, frames=max_iter,
                                interval=0,
                                repeat=False)
        plt.show()
        return self.__dict__


def load_show_image(path):
    with open(path) as file:
        lattice_dict = json.load(file)
        for array in lattice_dict['images']:
            fig, ax = plt.subplots()
            ax.imshow(array, cmap='jet')
            plt.show()


class TestTrainSetGenerator:
    def __init__(self, test_train_ratio=0.8):
        self.__test_train_ratio = test_train_ratio
        self.__images = []

    def write(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.__dict__, f)

    def load(self, fname):
        with open(fname) as f:
            self.__dict__ = json.load(f)

    def clean(self):
        self.__images = [i for i in self.__images if isinstance(i['image'], list)]

    def get_data(self):
        np.random.shuffle(self.__images)

        split_at = int(self.__test_train_ratio * len(self.__images))

        train_data = [t['image'] for t in self.__images[:split_at]]
        test_data = [t['image'] for t in self.__images[split_at:]]

        train_label = [t['label'] for t in self.__images[:split_at]]
        test_label = [t['label'] for t in self.__images[split_at:]]
        return (np.array(train_data, dtype=np.float32), np.array(train_label, dtype=np.int32)), (
            np.array(test_data, dtype=np.float32), np.array(test_label, dtype=np.int32))

    def add(self, images, temp, critical):
        for image in images:
            self.__images.append({'image': image, 'label': int(critical)})


if __name__ == '__main__':
    # np.seterr(all='raise')
    ttgen = TestTrainSetGenerator()
    kt = np.linspace(0.5, 5, 10)
    m = []
    E = []
    C_v = []
    chi = []
    for t in kt:
        print(f"\nIterating at temperature: {t}")
        ising = IsingLattice(50, t, t < T_CRIT)
        result_json = ising.start(500000, 0, 200000)
        ttgen.add(result_json['record_states'], t, result_json['critical'])
        m.append(np.abs(np.mean(result_json['magnetization'])))
        E.append(np.mean(result_json['energy']))
        C_v.append(np.var(result_json['energy']) / ((t ** 2) * 2500))
        chi.append(np.var(result_json['magnetization']) / t)
    ttgen.write(f"dump_testT00-1.json")
    plt.subplot(2, 2, 1)
    plt.title("Absolute Magnetization per spin")
    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--')
    plt.plot(kt, m)
    plt.subplot(2, 2, 2)
    plt.title("Energy per spin")
    plt.plot(kt, E)
    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--')
    plt.subplot(2, 2, 3)
    plt.title("Heat capacity")
    plt.plot(kt, C_v)
    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--')
    plt.subplot(2, 2, 4)
    plt.title("Susceptibility")
    plt.plot(kt, chi)
    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--')
    plt.show()
