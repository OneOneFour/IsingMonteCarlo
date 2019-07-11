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
    def __init__(self, width, height, kT, cold=True):
        self.width = width
        self.height = height
        self.kT = kT
        self.critical = bool(self.kT > 2 / np.log(1 + np.sqrt(2)))
        if cold:
            self.lattice = [[1 for i in range(self.width)] for j in range(self.height)]
        else:
            self.lattice = [[random.choice((-1, 1)) for i in range(self.width)] for j in range(self.height)]

        self.magnetization = [self.cur_magnetization()]
        self.energy = [self.energy_periodic()]
        self.record_states = []

    def energy_periodic(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):
                E += -1 * self.lattice[y][x] * (
                        self.lattice[(y + 1) % self.height][x] +
                        self.lattice[(y - 1) % self.height][x] +
                        self.lattice[y][(x + 1) % self.width] +
                        self.lattice[y][(x - 1) % self.width]
                )
        return E

    def energy_free_ends(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):  # Use nearest neighbouring 4 cells
                E += self.lattice[y][x] * (
                        (self.lattice[y + 1][x] if self.lattice[y + 1][x] < self.height else 0) +
                        (self.lattice[y - 1][x] if self.lattice[y - 1][x] >= 0 else 0) +
                        (self.lattice[y][x + 1] if self.lattice[y + 1][x] < self.width else 0) +
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
        return magnetization / (self.width * self.height)

        # Flip the configuration spin

    def __metropolis_step(self, f=0, im=None, max_iter=5000):
        if im:
            if f >= max_iter - 1:
                plt.close()
            # print(f"\rFrame:{f} of 5000", end='')
        rand_y, rand_x = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
        deltaE = 2 * self.lattice[rand_y][rand_x] * (
                self.lattice[(rand_y + 1) % self.height][rand_x] +
                self.lattice[(rand_y - 1) % self.height][rand_x] +
                self.lattice[rand_y][(rand_x + 1) % self.width] +
                self.lattice[rand_y][(rand_x - 1) % self.width]
        )
        r = random.random()
        w = np.exp((-1 / self.kT) * deltaE)
        if deltaE <= 0 or r <= w:
            self.lattice[rand_y][rand_x] *= -1
            self.magnetization.append(
                self.magnetization[-1] + 2 * self.lattice[rand_y][rand_x] / (self.width * self.height))
            self.energy.append(self.energy[-1] + deltaE)
        else:
            self.magnetization.append(self.magnetization[-1])
            self.energy.append(self.energy[-1])
        if im:
            im.set_data(self.lattice)
        return im,

    def start(self, max_iter=5000, export_every=0, delay=0):
        # corrcoeff = []
        for i in tqdm(range(max_iter)):
            self.__metropolis_step()
            if export_every != 0 and i >= delay:
                if i % export_every == 0:
                    # capture snapshot of the image
                    # if len(corrcoeff) > 0:
                    #     plt.plot(corrcoeff, label=f"{i}")
                    #     corrcoeff = []
                    self.record_states.append(pickle.loads(pickle.dumps(self.lattice)))
            #  if len(self.record_states) > 0:
            # corrcoeff.append(
            # np.corrcoef(np.array(self.lattice).flatten(), np.array(self.record_states[-1]).flatten())[0][1])

            # Check the correlation function
        self.energy = self.energy[delay:]
        self.magnetization = self.magnetization[delay:]
        # plt.title(f"R^2 correlation at T:{self.kT}")
        # plt.legend()
        # plt.show()

        return self.__dict__

    def start_animation(self, max_iter=500000):
        import matplotlib.animation as animation
        fig, ax = plt.subplots()

        im = plt.imshow(self.lattice, cmap='jet', animated=True, vmin=-1, vmax=1)

        animation.FuncAnimation(fig, self.__metropolis_step, fargs=(im, max_iter,), blit=True, frames=max_iter,
                                interval=0,
                                repeat=False)
        plt.show()
        return np.mean(self.magnetization), np.var(self.magnetization), np.mean(self.energy), np.var(self.energy)


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

    def get_data(self):
        np.random.shuffle(self.__images)

        split_at = int(self.__test_train_ratio * len(self.__images))
        train = self.__images[:split_at]
        test = self.__images[split_at:]
        return (np.array([t['image'] for t in train]), np.array([t['label'] for t in train])), (
            np.array([t['image'] for t in test]), np.array([t['label'] for t in test]))

    def add(self, images, temp, critical):
        for image in images:
            self.__images.append({'image': image, 'label': int(critical)})


if __name__ == '__main__':
    # np.seterr(all='raise')
    ttgen = TestTrainSetGenerator()
    kt = np.linspace(T_CRIT-0.1, T_CRIT + 0.1, 2)
    m = []
    E = []
    C_v = []
    chi = []
    for t in kt:
        print(f"\nIterating at temperature: {t}")
        ising = IsingLattice(50, 50, t)
        result_json = ising.start(100000000, 100000, 100000)
        ttgen.add(result_json['record_states'], t, result_json['critical'])
        m.append(np.abs(np.mean(result_json['magnetization'])))
        E.append(np.mean(result_json['energy']))
        C_v.append(np.var(result_json['energy']) / ((t ** 2) * 2500))
        chi.append(np.var(result_json['magnetization']) / t)
    ttgen.write(f"dump_testT0-1.json")
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
