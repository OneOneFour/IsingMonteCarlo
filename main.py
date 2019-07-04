import random
import numpy as np
import matplotlib.pyplot as plt
import copy


class IsingLattice:
    def __init__(self, width, height, kT, cold=True):
        self.width = width
        self.height = height
        self.kT = kT
        if cold:
            self.lattice = [[1 for i in range(self.width)] for j in range(self.height)]
        else:
            self.lattice = [[random.choice((-1, 1)) for i in range(self.width)] for j in range(self.height)]

        self.magnetization = []
        self.energy = []

    def energy_periodic(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):
                E += - 1 * self.lattice[y][x] * (
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

    def __metropolis_step(self):
        trial = copy.deepcopy(self)
        rand_x, rand_y = random.randint(0, 49), random.randint(0, 49)
        trial.lattice[rand_y][rand_x] *= -1  # Flip a single spin
        deltaE = trial.energy_periodic() - self.energy_periodic()
        if deltaE > 0:
            r = random.random()
            w = np.exp((-1 / self.kT) * deltaE)
            if r > w:
                return False
        self.lattice = trial.lattice
        self.magnetization.append(self.cur_magnetization())
        self.energy.append(self.energy_periodic())
        return True

    def start(self, max_iter=5000):
        for i in range(max_iter):
            self.__metropolis_step()
        return np.mean(self.magnetization), np.var(self.magnetization), np.mean(self.energy), np.var(self.energy)


if __name__ == '__main__':
    kt = np.linspace(0.1, 100, 25)
    m = []
    E = []
    C_v = []
    chi = []
    for t in kt:
        ising = IsingLattice(50, 50, t)
        m_bar, m_var, e_bar, e_var = ising.start()
        m.append(m_bar)
        E.append(e_bar)
        C_v.append(e_var / t ** 2)

    plt.plot(kt, m)
    plt.show()
    plt.plot(kt, E)
    plt.show()
    plt.plot(kt, C_v)
    plt.show()
