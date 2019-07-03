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

    def energy_periodic(self):
        E = 0
        for y in range(len(self.lattice)):
            for x in range(len(self.lattice[y])):
                E += self.lattice[y][x] * (
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

    def magnetization(self):
        magnetization = 0
        for row in self.lattice:
            for spin in row:
                magnetization += spin
        return magnetization

        # Flip the configuration spin


def metropolis_step(baseIsing):
    i=0
    while True:
        trial = copy.deepcopy(baseIsing)
        rand_x, rand_y = random.randint(0, 49), random.randint(0, 49)
        trial.lattice[rand_y][rand_x] *= -1
        deltaE = trial.energy_periodic() - baseIsing.energy_periodic()

        if deltaE <= 0:
            baseIsing = trial
        else:
            r = random.random()
            w = np.exp((-1 / baseIsing.kT) * deltaE)
            if r < w:
                baseIsing = trial

        i+=1
        if i % 200 == 0:
            baseIsing.showlattice()



if __name__ == '__main__':
    ising = IsingLattice(50, 50, 200)
    ising.showlattice()
    metropolis_step(ising)
