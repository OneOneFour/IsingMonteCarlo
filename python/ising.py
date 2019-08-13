import json
import random
from json import JSONDecodeError
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime as dt
import os

T_CRIT_ONS = 2 / np.log(1 + np.sqrt(2))


class IsingLattice:
    def __init__(self, size, kT, cold=True):
        self.size = size
        self.kT = kT
        self.critical = bool(self.kT > 2 / np.log(1 + np.sqrt(2)))
        if cold:
            self.lattice = np.array([[1 for i in range(self.size)] for j in range(self.size)])
        else:
            self.lattice = np.array([[random.choice((-1, 1)) for i in range(self.size)] for j in range(self.size)])

        self.record_states = []
        self.m = self.cur_magnetization(self.lattice, self.size)
        self.e = self.energy_periodic(self.lattice, self.size)

    @staticmethod
    def energy_periodic(lattice, size):
        E = 0
        for y in range(len(lattice)):
            for x in range(len(lattice[y])):
                E += -1 * lattice[y][x] * (
                        lattice[(y + 1) % size][x] +
                        lattice[(y - 1) % size][x] +
                        lattice[y][(x + 1) % size] +
                        lattice[y][(x - 1) % size]
                )
        return E

    @staticmethod
    def energy_free_ends(lattice, size):
        E = 0
        for y in range(len(lattice)):
            for x in range(len(lattice[y])):  # Use nearest neighbouring 4 cells
                E += lattice[y][x] * (
                        (lattice[y + 1][x] if lattice[y + 1][x] < size else 0) +
                        (lattice[y - 1][x] if lattice[y - 1][x] >= 0 else 0) +
                        (lattice[y][x + 1] if lattice[y + 1][x] < size else 0) +
                        (lattice[y][x - 1] if lattice[y + 1][x] >= 0 else 0)
                )
        return E

    def showlattice(self):
        fig, ax = plt.subplots()
        ax.imshow(self.lattice, cmap='jet')
        plt.show()

    @staticmethod
    def cur_magnetization(lattice, size):
        magnetization = 0
        for row in lattice:
            for spin in row:
                magnetization += spin
        return magnetization / (size ** 2)

    def abs_cur_magnetization(self):
        magnetization = 0
        for row in self.lattice:
            for spin in row:
                magnetization += spin
        return np.abs(magnetization / (self.size ** 2))
        # Flip the configuration spin

    def __wolff_step(self, i=0, im=None, max_iter=0):
        '''
        Work using bonds as opposed to individual spins on the lattice. Converts problem to a percolation problem
        which unlike Metropolis does not fall in to inf relaxation time at the critical temperature. :return: Artist
        object to be blitted to matplotlib canvas
        '''
        rand_y, rand_x = np.random.randint(0, self.size, size=2)
        cluster = [(rand_y, rand_x)]

        for p in cluster:
            for p_1 in [(p[0] - 1, p[1] - 1), (p[0] - 1, p[1] + 1), (p[0] + 1, p[1] - 1), (p[0] + 1, p[1] + 1)]:
                y_1, x_1 = (p_1[0] % self.size, p_1[1] % self.size)
                if (y_1, x_1) in cluster:
                    continue
                if self.lattice[p[0]][p[1]] == self.lattice[y_1][x_1]:
                    if random.random() <= (1 - np.exp(-2 / self.kT)):
                        cluster.append((y_1, x_1))

        self.lattice[np.transpose(cluster)] *= -1
        self.e = self.energy_periodic(self.lattice, self.size)
        self.m = self.cur_magnetization(self.lattice, self.size)
        if im:
            im.set_data(self.lattice)
            return im,
        else:
            return len(cluster) / self.size ** 2

    def __metropolis_step(self, i=0, im=None, max_iter=5000):
        for j in range(self.size ** 2):
            if im:
                if i >= max_iter - 1:
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
                self.m += (2 * self.lattice[rand_y][rand_x] / (self.size ** 2))
                self.e += deltaE

        if im:
            im.set_data(self.lattice)
            return im,
        return 1

    def start(self, method='metropolis', max_iter=5000, export_every=0, delay=0, log_correlation=False,
              animated_plots=False):
        if log_correlation:
            corrcoeff = []
        if export_every != 0:
            self.record_states = []
        i = 1
        sum_e = self.e
        sum_m = self.m
        sum_abs_m = abs(self.m)
        sum_esq = self.e * self.e
        sum_msq = self.m * self.m
        with tqdm(total=max_iter) as pbar:
            while i < max_iter:
                if method == 'metropolis':
                    steps = self.__metropolis_step(i=i)
                elif method == 'wolff':
                    steps = self.__wolff_step(i=i)
                else:
                    raise ValueError(
                        f"{method} is not a supported iteration method. Please choose from 'wolff' or 'metropolis' ")

                if export_every != 0 and i >= delay:
                    if int(i / export_every) >= len(self.record_states):
                        # capture snapshot of the image
                        if log_correlation:
                            if len(corrcoeff) > 0:
                                plt.plot(corrcoeff, label=f"{i}")
                                corrcoeff = []
                        self.record_states.append(self.lattice.tolist())
                if log_correlation:
                    if len(self.record_states) > 0:
                        a = np.array(self.lattice).flatten()
                        b = np.array(self.record_states[-1]).flatten()
                        c = np.corrcoef(a, b)
                        corrcoeff.append(c[0][1])

                # Update progress bar and loop progress
                i += steps
                pbar.update(steps)
                sum_e += self.e * steps
                sum_m += self.m * steps
                sum_abs_m += abs(self.m) * steps
                sum_esq += self.e * self.e * steps
                sum_msq += self.m * self.m * steps

        if log_correlation:
            plt.title(f"R^2 correlation at T:{self.kT}")
            plt.legend()
            plt.show()

        return sum_e / (max_iter), sum_abs_m / max_iter, sum_esq / (max_iter) - (
                sum_e / (max_iter)) ** 2, sum_msq / max_iter - (
                       sum_m / max_iter) ** 2

    def start_animation(self, method='metropolis', max_iter=500000):
        import matplotlib.animation as animation
        fig, ax = plt.subplots()

        im = plt.imshow(self.lattice, cmap='jet', animated=True, vmin=-1, vmax=1)

        animation.FuncAnimation(fig, self.__wolff_step if method == 'wolff' else self.__metropolis_step,
                                fargs=(im, max_iter), blit=True,
                                frames=max_iter,
                                interval=0,
                                repeat=False)
        plt.show()
        return None


def load_show_image(path):
    with open(path) as file:
        lattice_dict = json.load(file)
        for array in lattice_dict['images']:
            fig, ax = plt.subplots()
            ax.imshow(array, cmap='jet')
            plt.show()


class IsingData:
    def __init__(self, test_ratio=1, validation_ratio=1, train_ratio=1, size=50):
        self.__test_ratio = test_ratio
        self.__val_ratio = validation_ratio
        self.__train_ratio = train_ratio

        self.size = size
        self.__images = []

    @staticmethod
    def upgrade(dir):
        import glob
        os.chdir(dir)
        for file in glob.iglob('*.json'):
            obj = IsingData()
            try:
                with open(file, 'r') as f:
                    obj.__dict__ = json.load(f)
            except Exception as e:
                print(f"Exception in file: {file}")
                raise e
            finally:
                obj.write(file)

    def write(self, fname, batch_size=1):
        step = len(self.__images) / batch_size
        for i in range(batch_size):
            start = i * step
            end = (i + 1) * step
            with open(f"batch{i}_{fname}", 'w') as f:
                json.dump(self.__images[start:end], f)

    @staticmethod
    def autocorrect(fname):
        print(f"Attempting autocorrect on file {fname}")
        f = open(fname, 'r')
        contents = f.readlines()
        f.seek(0)
        try:
            json.load(f)
            f.close()
        except JSONDecodeError as json_err:
            f.close()
            error_pos = json_err.pos
            contents.insert(error_pos, ",")
            wf = open(fname, 'w')
            wf.write("".join(contents))
            wf.close()
            print(json_err)
            print("corrected")

    def load_json(self, json_f):
        import os
        with open(json_f, 'r') as meta_j:
            meta_json = json.load(meta_j)
            head, tail = os.path.split(json_f)
            os.chdir(head)
            if 'files' in meta_json:
                for f in meta_json['files']:
                    with open(f, 'r') as f:
                        self.__images.extend(json.load(f))
            else:
                raise ValueError("No files to run!")

    def load(self, fname):
        with open(fname, 'r') as f:
            try:
                inbound_dict = json.load(f)
            except JSONDecodeError as err:
                start, stop = max(0, err.pos - 20), min(err.pos + 20, len(err.doc))
                snippet = err.doc[start:stop]
                if err.pos < 20:
                    snippet = '... ' + snippet
                if err.pos + 20 < len(err.doc):
                    snippet += ' ...'
                print(snippet)
                raise err
            self.__images = inbound_dict

    def load_arr(self, files):
        for file in files:
            with open(file, 'r') as f:
                self.__images.extend(json.load(f))

    def clean(self):
        self.__images = [i for i in self.__images if isinstance(i['image'], list)]

    def get_data(self):
        np.random.shuffle(self.__images)
        # Training samples from 0 -> train_point
        train_point = int(self.__train_ratio * len(self.__images) / (
                self.__train_ratio + self.__test_ratio + self.__val_ratio))
        # Validaion samples from train_point -> val_point
        val_point = int(train_point + self.__val_ratio * len(self.__images) / (
                self.__train_ratio + self.__test_ratio + self.__val_ratio))
        # Test samples from val_point -> end

        train_data = [t['image'] for t in self.__images[:train_point]]
        validation_data = [t['image'] for t in self.__images[train_point:val_point]]
        test_data = [t['image'] for t in self.__images[val_point:]]

        train_label = [t['label'] for t in self.__images[:train_point]]
        validation_label = [t['label'] for t in self.__images[train_point:val_point]]
        test_label = [t['label'] for t in self.__images[val_point:]]

        return (np.array(train_data, dtype=np.float32), np.array(train_label, dtype=np.int32)), (
            np.array(test_data, dtype=np.float32), np.array(test_label, dtype=np.int32)), (
                   np.array(validation_data, dtype=np.float32), np.array(validation_label, dtype=np.int32))

    def plot_energy_spectrum(self, bins=10):
        if len(self.__images) == 0:
            raise ValueError("Please load images first! Call load,load_arr or load_json")
        # Sort by temperature

        temperatures = []
        for t in self.__images:
            if t['temp'] not in temperatures:
                temperatures.append(t['temp'])
        for temp in temperatures:
            energies = [IsingLattice.energy_periodic(t['image'], 50) for t in self.__images if t['temp'] == temp]
            plt.hist(energies, bins=bins, label=temp)
        plt.legend()
        plt.show()

    def get_data_flattened(self):
        np.random.shuffle(self.__images)
        # Training samples from 0 -> train_point
        train_point = int(self.__train_ratio * len(self.__images) / (
                self.__train_ratio + self.__test_ratio + self.__val_ratio))
        # Validaion samples from train_point -> val_point
        val_point = int(
            self.__val_ratio * len(self.__images) / (self.__train_ratio + self.__test_ratio + self.__val_ratio))
        # Test samples from val_point -> end

        train_data = [np.array(t['image']).flatten() for t in self.__images[:train_point]]
        validation_data = [np.array(t['image']).flatten() for t in self.__images[train_point:val_point]]
        test_data = [np.array(t['image']).flatten() for t in self.__images[val_point:]]

        train_label = [t['label'] for t in self.__images[:train_point]]
        validation_label = [t['label'] for t in self.__images[train_point:val_point]]
        test_label = [t['label'] for t in self.__images[val_point:]]

        return (np.array(train_data, dtype=np.float32), np.array(train_label, dtype=np.int32)), (
            np.array(test_data, dtype=np.float32), np.array(test_label, dtype=np.int32)), (
                   np.array(validation_data, dtype=np.float32), np.array(validation_label, dtype=np.int32))

    def add(self, images, temp, critical):
        for image in images:
            self.__images.append({'image': image, 'label': int(critical), 'temp': temp})


if __name__ == '__main__':
    # np.seterr(all='raise')

    size = 50
    ttgen = IsingData(size=size)
    min_res = 10 / (4 * size)
    kt = np.linspace(T_CRIT_ONS - min_res, T_CRIT_ONS + min_res, 2)

    M = []
    E = []
    C_v = []
    chi = []

    if sys.argv[1] == "animate":
        for t in kt:
            print(f"Iterating at temperature: {t}")
            ising = IsingLattice(size, t, False)
            ising.start_animation("metropolis", 10000)
    else:
        for t in kt:
            print(f"Iterating at temperature: {t}")
            ising = IsingLattice(size, t, t < T_CRIT_ONS)
            e, m, e_var, m_var = ising.start('wolff', 5000, 0, 0)
            # e,m,e_var,m_var = ising.start_animation('metropolis', 1000000)
            ttgen.add(ising.record_states, t, ising.critical)
            M.append(m)
            E.append(e)
            C_v.append(e_var / ((t ** 2) * 2500))
            chi.append(m_var / t)
        ttgen.write(f"dumps/{dt.now().strftime('%d-%m-%Y %H-%M-%S')}dump.json")
        plt.subplot(2, 2, 1)
        plt.title("Absolute Magnetization per spin")
        plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--')
        plt.plot(kt, M)
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
