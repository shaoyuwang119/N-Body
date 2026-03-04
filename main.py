import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NBody:
    G = 1
    N = 0
    mass = pos = vel = None
    rng = np.random.default_rng()
    scat = fig = ax = ani = None
    times=0

    def __init__(self, n):
        self.N = n
        self.mass = np.array(self.rng.uniform(low=100, high=200, size=n))
        self.pos = np.empty((n,2))
        self.vel = np.empty((n,2))
        for i in range(self.N):
            self.pos[i] = self.rng.uniform(low=-30.0, high=30.0, size=2)
            self.vel[i] = self.rng.uniform(low=-3.0, high=3.0, size=2)

        self.init_plot()

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.scat = plt.scatter(self.pos[:, 0], self.pos[:, 1], s=self.mass/2-40)

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)

    def step(self, t):
        for i in range(0, self.mass.size):
            net_force = np.zeros(2)
            m1 = self.mass[i]
            for j in range(0, self.mass.size):
                if i == j: continue
                m2 = self.mass[j]
                r = self.pos[j] - self.pos[i]
                r_mag = np.linalg.norm(r) + 0.5
                net_force += r * (self.G * m1 * m2 / r_mag ** 3)
            a = net_force / m1
            self.vel[i] += a * t
            self.pos[i] += self.vel[i] * t

    def update(self, frame):
        self.step(0.01)
        self.scat.set_offsets(self.pos)
        return [self.scat]

    def main(self):
        self.ani = FuncAnimation(plt.gcf(), func=self.update, frames=1000, interval=1, blit=True)
        plt.show()

nbody = NBody(20)
nbody.main()