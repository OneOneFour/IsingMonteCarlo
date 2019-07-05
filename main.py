import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from timeit import default_timer as timer

class IsingLattice:
    def __init__(self, width, height, kT, cold=True):
        self.width = width
        self.height = height
        self.kT = kT
        if cold:
            self.lattice = [[1 for i in range(self.width)] for j in range(self.height)]
        else:
            self.lattice = [[random.choice((-1, 1)) for i in range(self.width)] for j in range(self.height)]

        self.magnetization = [self.cur_magnetization()]
        self.energy = [self.energy_periodic()]

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
        rand_y,rand_x = random.randint(0,self.height -1),random.randint(0,self.width -1 )
        deltaE = 2*self.lattice[rand_y][rand_x]* (
                    self.lattice[(rand_y + 1) % self.height][rand_x] +
                    self.lattice[(rand_y - 1) % self.height][rand_x] +
                    self.lattice[rand_y][(rand_x + 1) % self.width] +
                    self.lattice[rand_y][(rand_x - 1) % self.width]
        )
        r = random.random()
        w = np.exp((-1/self.kT) * deltaE)
        if deltaE <=0 or r <= w:
            self.lattice[rand_y][rand_x] *= -1
            self.magnetization.append(self.magnetization[-1] + 2*self.lattice[rand_y][rand_x]/(self.width*self.height))
            self.energy.append(self.energy[-1] + deltaE)
        else:
            self.magnetization.append(self.magnetization[-1])
            self.energy.append(self.energy[-1])
        if im:
            im.set_data(self.lattice)
        return im,

    def start(self, max_iter=5000):
        for i in tqdm(range(max_iter)):
            self.__metropolis_step()
        return (np.mean(self.magnetization)), np.var(self.magnetization), np.mean(self.energy)/(self.width*self.height), np.var(self.energy)

    def start_anim(self, max_iter=5000):
        import matplotlib.animation as animation
        fig, ax = plt.subplots()

        im = plt.imshow(self.lattice, cmap='jet', animated=True, vmin=-1, vmax=1)

        animation.FuncAnimation(fig, self.__metropolis_step, fargs=(im, max_iter,), blit=True, frames=max_iter,
                                interval=0,
                                repeat=False)
        plt.show()
        return np.mean(self.magnetization), np.var(self.magnetization), np.mean(self.energy), np.var(self.energy)


if __name__ == '__main__':
    # np.seterr(all='raise')
    kt = np.linspace(1, 3, 50)
    m = []
    E = []
    C_v = []
    chi = []
    for t in kt:
        print(f"\nIterating at temperature: {t}")
        ising = IsingLattice(50, 50, t)
        m_bar, m_var, e_bar, e_var = ising.start(500000)
        m.append(m_bar)
        E.append(e_bar)
        C_v.append(e_var / t ** 2)

    plt.plot(kt, m)
    plt.show()
    plt.plot(kt, E)
    plt.show()
    plt.plot(kt, C_v)
    plt.show()
